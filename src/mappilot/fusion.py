"""Loose coupling of strapdown IMU propagation with external pose fixes.

:func:`fuse_imu_pose` walks a strictly increasing IMU timeline with a
constant-bias, left-endpoint kinematic propagation and, at IMU timestamps
carrying an external pose observation, right-composes a gain-scaled SE(3)
correction onto the predicted pose. The result is one immutable
:class:`FusionState` per IMU sample.

Conventions follow :mod:`mappilot.geometry` and :mod:`mappilot.inertial`
(right-handed coordinates; a pose maps body coordinates to world
coordinates):

* Each IMU sample is ``(timestamp, acceleration, angular_velocity)`` with
  the same record semantics as :func:`mappilot.inertial.preintegrate_imu`.
  At least two samples with strictly increasing timestamps are required.
* Each pose observation is ``(timestamp, pose)`` where ``pose`` is the
  observed body-to-world :class:`~mappilot.geometry.Pose3`. Observation
  timestamps must be strictly increasing, may be empty, and every
  timestamp must equal an IMU timestamp exactly; duplicate observations at
  one instant are forbidden.
* Between adjacent IMU times ``t[k]`` and ``t[k+1]`` the bias-corrected
  measurement of the *left* sample ``k`` is held constant. The pose
  propagates as

  ``R_new = R * Exp((omega - b_g) * dt)``

  and the corrected specific force is rotated with the rotation ``R`` at
  the beginning of the interval:

  ``a_world = R * (a - b_a) + gravity``
  ``p_new = p + v * dt + 0.5 * a_world * dt**2``
  ``v_new = v + a_world * dt``

* At an observation time the residual is
  ``error = T_pred.inverse().compose(T_obs)``. Its translation norm and the
  norm of the rotation part of ``Pose3.log(error)`` (its first three
  components) are tested against ``residual_threshold``; when a threshold
  is given, either norm strictly exceeding it rejects the observation
  (equality is accepted), leaving the predicted state untouched.
* An accepted observation scales the rotation and translation parts of
  ``Pose3.log(error)`` with ``rotation_gain`` and ``translation_gain``
  respectively; the ``Pose3.exp`` of the scaled tangent vector is
  right-composed onto the predicted pose. Velocity is never changed by an
  observation. An observation at the first IMU time corrects the initial
  state in exactly the same way.

This module is pure Python and depends only on the standard library plus
:mod:`mappilot.geometry`. Importing it never starts the HTTP service.

Validation policy mirrors :mod:`mappilot.inertial`: non-iterable samples or
records, malformed records, wrong vector dimensions, non-real scalars
(including booleans), a pose of the wrong type, or a threshold/gain of the
wrong type raise :class:`TypeError`; fewer than two IMU samples,
non-finite values, timestamps that are not strictly increasing in either
sequence, an observation timestamp that does not hit an IMU sample, a gain
outside ``[0, 1]``, a negative or non-finite threshold, and non-finite
propagation results raise :class:`ValueError`.
"""

from __future__ import annotations

import math
import numbers
from typing import Iterable, NamedTuple

from .geometry import Pose3

__all__ = ["FusionState", "fuse_imu_pose"]

_ZERO_VECTOR = (0.0, 0.0, 0.0)


class FusionState(NamedTuple):
    """Filter state at one IMU timestamp.

    Immutable. ``pose`` is the body-to-world :class:`~mappilot.geometry.Pose3`
    and ``velocity`` is the world-frame velocity after applying any
    observation at this timestamp. ``observation_used`` is ``None`` when no
    observation exists at this timestamp, ``True`` when one was applied,
    and ``False`` when one was present but rejected by the residual gate.
    """

    timestamp: float
    pose: Pose3
    velocity: tuple[float, float, float]
    observation_used: bool | None


def _real_scalar(value: object, name: str) -> float:
    """Coerce one real, non-boolean scalar to float (finiteness checked by caller)."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    return float(value)


def _finite_scalar(value: object, name: str) -> float:
    """Coerce one real, finite, non-boolean scalar to float."""
    result = _real_scalar(value, name)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {result!r}")
    return result


def _vector3(value: object, name: str) -> tuple[float, float, float]:
    """Validate one vector of exactly three finite, non-boolean real scalars."""
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be a sequence of 3 real numbers")
    try:
        items = tuple(value)  # type: ignore[call-overload]
    except TypeError:
        raise TypeError(f"{name} must be a sequence of 3 real numbers") from None
    if len(items) != 3:
        raise TypeError(f"{name} must have 3 elements, got {len(items)}")
    return (
        _finite_scalar(items[0], f"{name}[0]"),
        _finite_scalar(items[1], f"{name}[1]"),
        _finite_scalar(items[2], f"{name}[2]"),
    )


def _unpack_sample(
    sample: object, index: int
) -> tuple[float, tuple[float, float, float], tuple[float, float, float]]:
    """Validate one ``(timestamp, acceleration, angular_velocity)`` sample."""
    if isinstance(sample, (str, bytes, bytearray)):
        raise TypeError(
            f"imu_samples[{index}] must be a (timestamp, acceleration, "
            "angular_velocity) triple"
        )
    try:
        items = tuple(sample)  # type: ignore[call-overload]
    except TypeError:
        raise TypeError(
            f"imu_samples[{index}] must be a (timestamp, acceleration, "
            "angular_velocity) triple"
        ) from None
    if len(items) != 3:
        raise TypeError(
            f"imu_samples[{index}] must have exactly 3 elements "
            f"(timestamp, acceleration, angular_velocity), got {len(items)}"
        )
    timestamp = _finite_scalar(items[0], f"imu_samples[{index}][0]")
    acceleration = _vector3(items[1], f"imu_samples[{index}][1]")
    angular_velocity = _vector3(items[2], f"imu_samples[{index}][2]")
    return timestamp, acceleration, angular_velocity


def _unpack_observation(
    observation: object, index: int
) -> tuple[float, Pose3]:
    """Validate one ``(timestamp, pose)`` observation."""
    if isinstance(observation, (str, bytes, bytearray)):
        raise TypeError(
            f"pose_observations[{index}] must be a (timestamp, pose) pair"
        )
    try:
        items = tuple(observation)  # type: ignore[call-overload]
    except TypeError:
        raise TypeError(
            f"pose_observations[{index}] must be a (timestamp, pose) pair"
        ) from None
    if len(items) != 2:
        raise TypeError(
            f"pose_observations[{index}] must have exactly 2 elements "
            f"(timestamp, pose), got {len(items)}"
        )
    timestamp = _finite_scalar(items[0], f"pose_observations[{index}][0]")
    pose = items[1]
    if not isinstance(pose, Pose3):
        raise TypeError(
            f"pose_observations[{index}][1] must be a Pose3, got "
            f"{type(pose).__name__}"
        )
    return timestamp, pose


def _gain(value: object, name: str) -> float:
    """Validate one correction gain: a finite real in [0, 1]."""
    result = _real_scalar(value, name)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {result!r}")
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1], got {result!r}")
    return result


def _threshold(value: object) -> float | None:
    """Validate an optional residual gate: ``None`` or a finite non-negative real."""
    if value is None:
        return None
    result = _real_scalar(value, "residual_threshold")
    if not math.isfinite(result):
        raise ValueError(f"residual_threshold must be finite, got {result!r}")
    if result < 0.0:
        raise ValueError(
            f"residual_threshold must be non-negative, got {result!r}"
        )
    return result


def _finite_state(
    translation: tuple[float, float, float],
    quaternion: tuple[float, float, float, float],
    velocity: tuple[float, float, float],
) -> None:
    for component in (*translation, *quaternion, *velocity):
        if not math.isfinite(component):
            raise ValueError("IMU propagation produced non-finite state")


def fuse_imu_pose(
    imu_samples: Iterable[
        tuple[float, tuple[float, float, float], tuple[float, float, float]]
    ],
    pose_observations: Iterable[tuple[float, Pose3]],
    initial_pose: Pose3,
    initial_velocity: Iterable[float],
    accelerometer_bias: Iterable[float] = _ZERO_VECTOR,
    gyroscope_bias: Iterable[float] = _ZERO_VECTOR,
    gravity: Iterable[float] = _ZERO_VECTOR,
    translation_gain: float = 1.0,
    rotation_gain: float = 1.0,
    residual_threshold: float | None = None,
) -> tuple[FusionState, ...]:
    """Fuse strapdown IMU propagation with timestamped pose observations.

    Parameters
    ----------
    imu_samples:
        ``(timestamp, acceleration, angular_velocity)`` records with
        strictly increasing absolute timestamps (seconds); at least two
        samples. The iterable is consumed exactly once.
    pose_observations:
        ``(timestamp, pose)`` pairs ordered by strictly increasing
        timestamps; the sequence may be empty. Each timestamp must equal an
        IMU timestamp exactly and no timestamp may repeat. The iterable is
        consumed exactly once.
    initial_pose:
        Body-to-world :class:`~mappilot.geometry.Pose3` at the first IMU
        timestamp.
    initial_velocity:
        World-frame velocity 3-vector at the first IMU timestamp.
    accelerometer_bias, gyroscope_bias:
        Constant body-frame 3-vector biases subtracted from the
        acceleration and angular-rate measurements; zero by default.
    gravity:
        World-frame gravity 3-vector added to the rotated specific force;
        zero by default.
    translation_gain, rotation_gain:
        Gains in ``[0, 1]`` scaling the translation and rotation parts of
        each applied correction; 1.0 by default (full correction).
    residual_threshold:
        ``None`` (default, no gating) or a non-negative finite bound; an
        observation is rejected when the residual translation norm or
        rotation norm is strictly greater than it. Equality is accepted.

    Returns
    -------
    tuple[FusionState, ...]
        One immutable :class:`FusionState` per IMU sample, in timestamp
        order, each carrying the pose (and world velocity) after any
        observation at that timestamp. ``observation_used`` is ``None``
        without an observation, ``True`` when one was applied, and
        ``False`` when the gate rejected one.

    Raises
    ------
    TypeError
        Non-iterable inputs or records, malformed records, wrong vector
        dimensions, non-real or boolean values, an ``initial_pose`` or
        observed pose that is not a :class:`~mappilot.geometry.Pose3`, or
        gains/threshold of the wrong numeric type.
    ValueError
        Fewer than two IMU samples; non-finite values; timestamps that are
        not strictly increasing in either sequence; an observation
        timestamp missing from the IMU timeline; a gain outside ``[0, 1]``;
        a negative or non-finite threshold; or non-finite propagation
        results.
    """
    if not isinstance(initial_pose, Pose3):
        raise TypeError(
            f"initial_pose must be a Pose3, got {type(initial_pose).__name__}"
        )
    velocity0 = _vector3(initial_velocity, "initial_velocity")
    accel_bias = _vector3(accelerometer_bias, "accelerometer_bias")
    gyro_bias = _vector3(gyroscope_bias, "gyroscope_bias")
    gravity_vector = _vector3(gravity, "gravity")
    t_gain = _gain(translation_gain, "translation_gain")
    r_gain = _gain(rotation_gain, "rotation_gain")
    threshold = _threshold(residual_threshold)

    # Materialize both iterables exactly once; validated records are freshly
    # built floats and Pose3 values are immutable, so caller objects are
    # never mutated or aliased into mutable state.
    if isinstance(imu_samples, (str, bytes, bytearray)):
        raise TypeError(
            "imu_samples must be an iterable of (timestamp, acceleration, "
            "angular_velocity) triples"
        )
    try:
        raw_samples = list(imu_samples)  # type: ignore[call-overload]
    except TypeError:
        raise TypeError(
            "imu_samples must be an iterable of (timestamp, acceleration, "
            "angular_velocity) triples"
        ) from None
    if len(raw_samples) < 2:
        raise ValueError(
            f"imu_samples must contain at least two samples, got {len(raw_samples)}"
        )
    records = [_unpack_sample(sample, index) for index, sample in enumerate(raw_samples)]

    if isinstance(pose_observations, (str, bytes, bytearray)):
        raise TypeError(
            "pose_observations must be an iterable of (timestamp, pose) pairs"
        )
    try:
        raw_observations = list(pose_observations)  # type: ignore[call-overload]
    except TypeError:
        raise TypeError(
            "pose_observations must be an iterable of (timestamp, pose) pairs"
        ) from None
    observations = [
        _unpack_observation(observation, index)
        for index, observation in enumerate(raw_observations)
    ]

    previous_time: float | None = None
    for index, (timestamp, _, _) in enumerate(records):
        if previous_time is not None and timestamp <= previous_time:
            raise ValueError(
                "imu_samples timestamps must be strictly increasing, got "
                f"{previous_time!r} followed by {timestamp!r}"
            )
        previous_time = timestamp

    previous_obs_time: float | None = None
    for index, (timestamp, _) in enumerate(observations):
        if previous_obs_time is not None and timestamp <= previous_obs_time:
            raise ValueError(
                "pose_observations timestamps must be strictly increasing, "
                f"got {previous_obs_time!r} followed by {timestamp!r}"
            )
        previous_obs_time = timestamp

    imu_times = {timestamp for timestamp, _, _ in records}
    for timestamp, _ in observations:
        if timestamp not in imu_times:
            raise ValueError(
                f"pose observation timestamp {timestamp!r} does not match "
                "any IMU sample timestamp"
            )
    observations_by_time = {
        timestamp: observed_pose for timestamp, observed_pose in observations
    }

    def apply_observation(
        predicted: Pose3, observed: Pose3
    ) -> tuple[Pose3, bool]:
        """Right-compose the gain-scaled residual, honoring the gate."""
        error = predicted.inverse().compose(observed)
        xi = error.log()
        if threshold is not None:
            translation_norm = math.hypot(*error.translation)
            rotation_norm = math.hypot(xi[0], xi[1], xi[2])
            if translation_norm > threshold or rotation_norm > threshold:
                return predicted, False
        scaled = (
            r_gain * xi[0],
            r_gain * xi[1],
            r_gain * xi[2],
            t_gain * xi[3],
            t_gain * xi[4],
            t_gain * xi[5],
        )
        return predicted.compose(Pose3.exp(scaled)), True

    states: list[FusionState] = []

    pose = initial_pose
    velocity = velocity0
    first_timestamp = records[0][0]
    first_observation = observations_by_time.get(first_timestamp)
    if first_observation is not None:
        # An observation at the first IMU time corrects the initial state;
        # the gate applies exactly as at any other time.
        pose, used = apply_observation(pose, first_observation)
        used_flag: bool | None = used
    else:
        used_flag = None
    states.append(FusionState(first_timestamp, pose, velocity, used_flag))

    for index in range(1, len(records)):
        timestamp, acceleration, angular_velocity = records[index - 1]
        next_timestamp = records[index][0]
        dt = next_timestamp - timestamp

        corrected_accel = (
            acceleration[0] - accel_bias[0],
            acceleration[1] - accel_bias[1],
            acceleration[2] - accel_bias[2],
        )
        corrected_omega = (
            angular_velocity[0] - gyro_bias[0],
            angular_velocity[1] - gyro_bias[1],
            angular_velocity[2] - gyro_bias[2],
        )

        # Left-endpoint convention: the specific force is rotated with the
        # rotation at the beginning of the interval (rotation only -- the
        # pose translation must not enter the acceleration), while the
        # angular rate advances the rotation via right composition.
        r00, r01, r02 = pose.rotation[0]
        r10, r11, r12 = pose.rotation[1]
        r20, r21, r22 = pose.rotation[2]
        ax, ay, az = corrected_accel
        rotated_accel = (
            r00 * ax + r01 * ay + r02 * az,
            r10 * ax + r11 * ay + r12 * az,
            r20 * ax + r21 * ay + r22 * az,
        )
        accel_world = (
            rotated_accel[0] + gravity_vector[0],
            rotated_accel[1] + gravity_vector[1],
            rotated_accel[2] + gravity_vector[2],
        )
        half_dt_squared = 0.5 * dt * dt
        translation = pose.translation
        new_translation = (
            translation[0] + velocity[0] * dt + accel_world[0] * half_dt_squared,
            translation[1] + velocity[1] * dt + accel_world[1] * half_dt_squared,
            translation[2] + velocity[2] * dt + accel_world[2] * half_dt_squared,
        )
        new_velocity = (
            velocity[0] + accel_world[0] * dt,
            velocity[1] + accel_world[1] * dt,
            velocity[2] + accel_world[2] * dt,
        )
        rotation_step = Pose3.exp(
            (
                corrected_omega[0] * dt,
                corrected_omega[1] * dt,
                corrected_omega[2] * dt,
                0.0,
                0.0,
                0.0,
            )
        )
        new_rotation_pose = pose.compose(rotation_step)
        _finite_state(
            new_translation, new_rotation_pose.quaternion, new_velocity
        )
        pose = Pose3(new_translation, new_rotation_pose.quaternion)
        velocity = new_velocity

        observation = observations_by_time.get(next_timestamp)
        if observation is None:
            used_flag = None
        else:
            pose, used = apply_observation(pose, observation)
            used_flag = used
        states.append(FusionState(next_timestamp, pose, velocity, used_flag))

    return tuple(states)
