"""Fuse inertial prediction with external body-to-world pose observations.

:func:`fuse_imu_pose` runs a dead-reckoning IMU propagator and, at selected
sample times, corrects the predicted :class:`~mappilot.geometry.Pose3`
toward an externally observed pose. Unlike pre-integration
(:mod:`mappilot.inertial`), the state lives in the world frame and gravity
is applied explicitly.

Conventions follow :mod:`mappilot.geometry` (right-handed coordinates):

* Each IMU sample is ``(timestamp, acceleration, angular_velocity)``; the
  timestamp is an absolute time in seconds. Acceleration is specific force
  and angular velocity is body rate, both measured in the body frame.
* Timestamps must be strictly increasing and at least two samples are
  required. For the interval ``[t[k], t[k+1]]`` the bias-corrected
  measurement of the *left* sample ``k`` is held constant.
* Propagation uses

  ``R_new = R * Exp((omega - b_g) * dt)`` (right composition),
  ``a_world = R * (a - b_a) + gravity``,
  ``p_new = p + v*dt + 0.5*a_world*dt**2``,
  ``v_new = v + a_world*dt``.

* A pose observation is ``(timestamp, pose)`` with strictly increasing
  timestamps; every timestamp must equal an IMU sample timestamp exactly
  and no two observations may share a timestamp. The sequence may be empty.
* At an observation time the residual is
  ``error = T_pred.inverse().compose(T_obs)``. Its translation norm and the
  norm of the rotation part (first three components of ``Pose3.log``) are
  compared against ``residual_threshold``; if either norm is *strictly*
  greater than the threshold the prediction is kept and the observation is
  marked rejected (equality is accepted). Otherwise the tangent vector is
  scaled component-wise by ``rotation_gain`` / ``translation_gain``, mapped
  through ``Pose3.exp`` and right-composed onto the prediction. Velocity is
  never corrected directly. An observation at the first IMU timestamp
  corrects the initial state in the same way.

This module is pure Python and depends only on the standard library plus
:mod:`mappilot.geometry`. Importing it never starts the HTTP service.

Validation policy mirrors :mod:`mappilot.geometry` and
:mod:`mappilot.inertial`: non-iterable records, malformed shapes, wrong
vector dimensions, non-real scalars (including booleans), and observations
whose pose is not a :class:`~mappilot.geometry.Pose3` raise
:class:`TypeError`; fewer than two IMU samples, non-finite values,
duplicate or out-of-order timestamps in either input, observation
timestamps that do not hit an IMU sample, gains outside ``[0, 1]``, a
negative or non-finite threshold, and non-finite propagation results raise
:class:`ValueError`.
"""

from __future__ import annotations

import math
import numbers
from typing import Iterable, NamedTuple

from .geometry import Pose3

__all__ = ["FusionState", "fuse_imu_pose"]

_ZERO_VECTOR = (0.0, 0.0, 0.0)


class FusionState(NamedTuple):
    """One fused state at one IMU timestamp.

    Immutable. ``pose`` maps body to world coordinates and ``velocity`` is
    the world-frame velocity. ``observation_used`` is ``None`` when no
    observation existed at this timestamp, ``True`` when one was applied,
    and ``False`` when one was rejected by the residual gate.
    """

    timestamp: float
    pose: Pose3
    velocity: tuple[float, float, float]
    observation_used: bool | None


def _finite_scalar(value: object, name: str) -> float:
    """Coerce one real, finite, non-boolean scalar to float."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    result = float(value)
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
            f"pose_observations[{index}][1] must be a Pose3, "
            f"got {type(pose).__name__}"
        )
    return timestamp, pose


def _gain(value: object, name: str) -> float:
    """Validate a correction gain: real, non-boolean, finite, in [0, 1]."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or result > 1.0:
        raise ValueError(f"{name} must lie in [0, 1], got {result!r}")
    return result


def _threshold(value: object) -> float | None:
    """Validate the optional residual gate: None or a finite, non-negative scalar."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(
            f"residual_threshold must be None or a real number, "
            f"got {type(value).__name__}"
        )
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(
            f"residual_threshold must be finite and non-negative, got {result!r}"
        )
    return result


def _finite_pose(pose: Pose3) -> None:
    if not all(math.isfinite(component) for component in pose.translation):
        raise ValueError("IMU propagation produced non-finite position")
    if not all(math.isfinite(component) for component in pose.quaternion):
        raise ValueError("IMU propagation produced non-finite rotation")


def _finite_velocity(velocity: tuple[float, float, float]) -> None:
    if not all(math.isfinite(component) for component in velocity):
        raise ValueError("IMU propagation produced non-finite velocity")


def _materialize(iterable: object, what: str) -> list:
    """Consume one input iterable exactly once into a fresh list."""
    if isinstance(iterable, (str, bytes, bytearray)):
        raise TypeError(f"{what} must be an iterable")
    try:
        return list(iterable)  # type: ignore[call-overload]
    except TypeError:
        raise TypeError(f"{what} must be an iterable") from None


def _apply_observation(
    predicted: Pose3,
    observed: Pose3,
    translation_gain: float,
    rotation_gain: float,
    residual_threshold: float | None,
) -> tuple[Pose3, bool]:
    """Correct ``predicted`` toward ``observed``; return (pose, accepted)."""
    error = predicted.inverse().compose(observed)
    xi = error.log()
    if residual_threshold is not None:
        translation_norm = math.hypot(*error.translation)
        rotation_norm = math.hypot(xi[0], xi[1], xi[2])
        # Strictly greater rejects; equality is accepted.
        if translation_norm > residual_threshold or rotation_norm > residual_threshold:
            return predicted, False
    increment = Pose3.exp(
        (
            rotation_gain * xi[0],
            rotation_gain * xi[1],
            rotation_gain * xi[2],
            translation_gain * xi[3],
            translation_gain * xi[4],
            translation_gain * xi[5],
        )
    )
    return predicted.compose(increment), True


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
    """Fuse IMU propagation with timestamped external pose observations.

    Parameters
    ----------
    imu_samples:
        ``(timestamp, acceleration, angular_velocity)`` records with strictly
        increasing absolute timestamps (seconds). At least two samples are
        required. The iterable is consumed exactly once.
    pose_observations:
        ``(timestamp, pose)`` pairs with strictly increasing timestamps;
        each timestamp must equal an IMU sample timestamp exactly. May be
        empty. Consumed exactly once.
    initial_pose:
        Body-to-world :class:`~mappilot.geometry.Pose3` at the first IMU
        timestamp.
    initial_velocity:
        World-frame velocity ``(vx, vy, vz)`` at the first IMU timestamp.
    accelerometer_bias:
        Constant 3-axis accelerometer bias subtracted from every
        acceleration sample; defaults to zero.
    gyroscope_bias:
        Constant 3-axis gyroscope bias subtracted from every angular-rate
        sample; defaults to zero.
    gravity:
        World-frame gravity vector added to the rotated specific force;
        defaults to zero.
    translation_gain:
        Gain in ``[0, 1]`` scaling the translation part of the observation
        residual; defaults to ``1.0`` (full correction).
    rotation_gain:
        Gain in ``[0, 1]`` scaling the rotation part of the residual;
        defaults to ``1.0``.
    residual_threshold:
        ``None`` (default, every in-window observation is applied) or a
        non-negative finite scalar; an observation whose residual
        translation norm or rotation-angle norm is strictly greater than it
        is rejected. Equality is accepted.

    Returns
    -------
    tuple[FusionState, ...]
        One immutable :class:`FusionState` per IMU timestamp, in order.
        ``observation_used`` is ``None`` without an observation, ``True``
        when applied, and ``False`` when gate-rejected. An observation at
        the first timestamp corrects the initial state.

    Raises
    ------
    TypeError
        Non-iterable records, malformed shapes, wrong vector dimensions,
        non-real or boolean numeric values, a non-:class:`Pose3`
        observation or initial pose, non-numeric gains, or a threshold
        that is neither ``None`` nor a real number.
    ValueError
        Fewer than two IMU samples; non-finite values; duplicate or
        out-of-order timestamps in either input; observation timestamps
        that do not match an IMU timestamp; gains outside ``[0, 1]``; a
        negative or non-finite threshold; or non-finite propagation
        results.
    """
    if not isinstance(initial_pose, Pose3):
        raise TypeError(
            f"initial_pose must be a Pose3, got {type(initial_pose).__name__}"
        )
    velocity = _vector3(initial_velocity, "initial_velocity")
    accel_bias = _vector3(accelerometer_bias, "accelerometer_bias")
    gyro_bias = _vector3(gyroscope_bias, "gyroscope_bias")
    gravity_vector = _vector3(gravity, "gravity")
    t_gain = _gain(translation_gain, "translation_gain")
    r_gain = _gain(rotation_gain, "rotation_gain")
    threshold = _threshold(residual_threshold)

    raw_samples = _materialize(imu_samples, "imu_samples")
    if len(raw_samples) < 2:
        raise ValueError(
            f"imu_samples must contain at least two samples, got {len(raw_samples)}"
        )
    records = [_unpack_sample(sample, index) for index, sample in enumerate(raw_samples)]

    previous_time: float | None = None
    for index, (timestamp, _, _) in enumerate(records):
        if previous_time is not None and timestamp <= previous_time:
            raise ValueError(
                "imu sample timestamps must be strictly increasing, got "
                f"{previous_time!r} followed by {timestamp!r}"
            )
        previous_time = timestamp

    raw_observations = _materialize(pose_observations, "pose_observations")
    observations = [
        _unpack_observation(observation, index)
        for index, observation in enumerate(raw_observations)
    ]
    previous_time = None
    for timestamp, _ in observations:
        if previous_time is not None and timestamp <= previous_time:
            raise ValueError(
                "pose observation timestamps must be strictly increasing, got "
                f"{previous_time!r} followed by {timestamp!r}"
            )
        previous_time = timestamp

    sample_times = {timestamp for timestamp, _, _ in records}
    observations_by_time: dict[float, Pose3] = {}
    for timestamp, observed_pose in observations:
        if timestamp not in sample_times:
            raise ValueError(
                f"pose observation timestamp {timestamp!r} does not match any "
                "IMU sample timestamp"
            )
        observations_by_time[timestamp] = observed_pose

    pose = initial_pose
    states: list[FusionState] = []

    first_timestamp = records[0][0]
    observation_used: bool | None = None
    if first_timestamp in observations_by_time:
        pose, accepted = _apply_observation(
            pose,
            observations_by_time[first_timestamp],
            t_gain,
            r_gain,
            threshold,
        )
        observation_used = accepted
    states.append(FusionState(first_timestamp, pose, velocity, observation_used))

    for index in range(len(records) - 1):
        timestamp, acceleration, angular_velocity = records[index]
        next_timestamp = records[index + 1][0]
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

        # Left-end hold: world acceleration rotates the corrected specific
        # force by R(t[k]) (translation of the pose is not applied).
        rotation = pose.rotation
        rotated_accel = (
            rotation[0][0] * corrected_accel[0]
            + rotation[0][1] * corrected_accel[1]
            + rotation[0][2] * corrected_accel[2],
            rotation[1][0] * corrected_accel[0]
            + rotation[1][1] * corrected_accel[1]
            + rotation[1][2] * corrected_accel[2],
            rotation[2][0] * corrected_accel[0]
            + rotation[2][1] * corrected_accel[1]
            + rotation[2][2] * corrected_accel[2],
        )
        a_world = (
            rotated_accel[0] + gravity_vector[0],
            rotated_accel[1] + gravity_vector[1],
            rotated_accel[2] + gravity_vector[2],
        )
        position = pose.translation
        new_position = (
            position[0] + velocity[0] * dt + 0.5 * a_world[0] * dt * dt,
            position[1] + velocity[1] * dt + 0.5 * a_world[1] * dt * dt,
            position[2] + velocity[2] * dt + 0.5 * a_world[2] * dt * dt,
        )
        velocity = (
            velocity[0] + a_world[0] * dt,
            velocity[1] + a_world[1] * dt,
            velocity[2] + a_world[2] * dt,
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
        pose = Pose3(new_position, pose.compose(rotation_step).quaternion)

        _finite_pose(pose)
        _finite_velocity(velocity)

        observation_used = None
        if next_timestamp in observations_by_time:
            pose, accepted = _apply_observation(
                pose,
                observations_by_time[next_timestamp],
                t_gain,
                r_gain,
                threshold,
            )
            observation_used = accepted
        states.append(
            FusionState(next_timestamp, pose, velocity, observation_used)
        )

    return tuple(states)
