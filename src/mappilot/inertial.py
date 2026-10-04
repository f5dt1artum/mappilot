"""IMU pre-integration: compress a timestamped IMU window into one increment.

Visual or LiDAR back-ends optimize poses at keyframe rate, far below IMU
rate. :func:`preintegrate_imu` folds every sample between two keyframes into
a single relative motion: a :class:`~mappilot.geometry.Pose3`, a velocity
increment, and the covered duration. Gravity and the Earth frame never
enter the computation; only body-frame specific force and angular rate are
integrated, so the output can be reused by any back-end.

Conventions follow :mod:`mappilot.geometry` (right-handed coordinates):

* Each sample is ``(timestamp, acceleration, angular_velocity)``; the
  timestamp is an absolute time in seconds and the two 3-vectors are the
  specific force and angular velocity measured in the body frame at that
  instant.
* Timestamps must be strictly increasing. For the interval
  ``[t[k], t[k+1]]`` the bias-corrected measurement of the *left* sample
  ``k`` is held constant; biases are constant over the whole window.
* Integration starts from identity rotation, zero velocity, zero position.
  For interval length ``dt`` the corrected specific force is first rotated
  into the starting body frame with the accumulated incremental rotation,
  then ``dp += dv*dt + 0.5*a*dt**2`` and ``dv += a*dt`` are applied, and
  finally the incremental rotation is right-composed with
  ``Pose3.exp((wx*dt, wy*dt, wz*dt, 0, 0, 0))``.

This module is pure Python and depends only on the standard library plus
:mod:`mappilot.geometry`. Importing it never starts the HTTP service.

Validation policy mirrors :mod:`mappilot.geometry`: non-iterable samples or
vectors, malformed records, wrong vector dimensions, and non-real scalars
(including booleans) raise :class:`TypeError`; fewer than two samples,
non-finite values, duplicate or out-of-order timestamps, a non-positive or
non-finite ``max_interval``, an interval strictly longer than
``max_interval``, and non-finite accumulation results raise
:class:`ValueError`.
"""

from __future__ import annotations

import math
import numbers
from typing import Iterable, NamedTuple

from .geometry import Pose3

__all__ = ["PreintegratedImu", "preintegrate_imu"]

_ZERO_VECTOR = (0.0, 0.0, 0.0)


class PreintegratedImu(NamedTuple):
    """Result of :func:`preintegrate_imu`.

    Immutable. ``delta_pose`` holds the accumulated position increment as
    its translation and the accumulated incremental rotation; velocities
    are expressed in the starting body frame. ``duration`` is the difference
    between the last and first timestamps and ``intervals`` is the number of
    integrated intervals (samples minus one).
    """

    delta_pose: Pose3
    delta_velocity: tuple[float, float, float]
    duration: float
    intervals: int


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
            f"samples[{index}] must be a (timestamp, acceleration, "
            "angular_velocity) triple"
        )
    try:
        items = tuple(sample)  # type: ignore[call-overload]
    except TypeError:
        raise TypeError(
            f"samples[{index}] must be a (timestamp, acceleration, "
            "angular_velocity) triple"
        ) from None
    if len(items) != 3:
        raise TypeError(
            f"samples[{index}] must have exactly 3 elements "
            f"(timestamp, acceleration, angular_velocity), got {len(items)}"
        )
    timestamp = _finite_scalar(items[0], f"samples[{index}][0]")
    acceleration = _vector3(items[1], f"samples[{index}][1]")
    angular_velocity = _vector3(items[2], f"samples[{index}][2]")
    return timestamp, acceleration, angular_velocity


def _finite_vector3(value: tuple[float, float, float], what: str) -> None:
    if not all(math.isfinite(component) for component in value):
        raise ValueError(f"pre-integration produced non-finite {what}")


def preintegrate_imu(
    samples: Iterable[tuple[float, tuple[float, float, float], tuple[float, float, float]]],
    accelerometer_bias: Iterable[float] = _ZERO_VECTOR,
    gyroscope_bias: Iterable[float] = _ZERO_VECTOR,
    max_interval: float | None = None,
) -> PreintegratedImu:
    """Integrate a time-ordered IMU window into one relative increment.

    Parameters
    ----------
    samples:
        ``(timestamp, acceleration, angular_velocity)`` records with strictly
        increasing absolute timestamps (seconds). At least two samples are
        required. The iterable is consumed exactly once.
    accelerometer_bias:
        Constant 3-axis accelerometer bias subtracted from every
        acceleration sample; defaults to the zero vector.
    gyroscope_bias:
        Constant 3-axis gyroscope bias subtracted from every angular-rate
        sample; defaults to the zero vector.
    max_interval:
        ``None`` (default) or a positive finite number bounding the time
        difference between adjacent samples; an interval strictly longer
        than this raises :class:`ValueError`, while equality is accepted.

    Returns
    -------
    PreintegratedImu
        Immutable result with ``delta_pose`` (translation ``dp``, rotation
        ``dR``), ``delta_velocity``, ``duration`` (last minus first
        timestamp), and ``intervals`` (number of samples minus one).

    Raises
    ------
    TypeError
        Non-iterable samples or vectors, malformed records, wrong vector
        dimensions, non-real or boolean measurements/biases, or a
        ``max_interval`` that is neither ``None`` nor a real number.
    ValueError
        Fewer than two samples; non-finite values; duplicate or
        out-of-order timestamps; a non-positive or non-finite
        ``max_interval``; an interval exceeding it; or non-finite
        accumulation results.
    """
    accel_bias = _vector3(accelerometer_bias, "accelerometer_bias")
    gyro_bias = _vector3(gyroscope_bias, "gyroscope_bias")

    if max_interval is not None:
        interval_limit = _real_scalar(max_interval, "max_interval")
        if not math.isfinite(interval_limit):
            raise ValueError(
                f"max_interval must be finite when given, got {interval_limit!r}"
            )
        if interval_limit <= 0.0:
            raise ValueError(
                f"max_interval must be positive when given, got {interval_limit!r}"
            )
    else:
        interval_limit = None

    # Materialize the samples exactly once; validated tuples are freshly
    # built floats, so caller objects are never mutated or retained.
    if isinstance(samples, (str, bytes, bytearray)):
        raise TypeError(
            "samples must be an iterable of (timestamp, acceleration, "
            "angular_velocity) triples"
        )
    try:
        raw_samples = list(samples)  # type: ignore[call-overload]
    except TypeError:
        raise TypeError(
            "samples must be an iterable of (timestamp, acceleration, "
            "angular_velocity) triples"
        ) from None
    if len(raw_samples) < 2:
        raise ValueError(
            f"samples must contain at least two samples, got {len(raw_samples)}"
        )
    records = [
        _unpack_sample(sample, index) for index, sample in enumerate(raw_samples)
    ]

    previous_time: float | None = None
    for index, (timestamp, _, _) in enumerate(records):
        if previous_time is not None:
            if timestamp <= previous_time:
                raise ValueError(
                    "sample timestamps must be strictly increasing, got "
                    f"{previous_time!r} followed by {timestamp!r}"
                )
            dt = timestamp - previous_time
            if interval_limit is not None and dt > interval_limit:
                raise ValueError(
                    f"interval between samples {index - 1} and {index} is "
                    f"{dt!r}, exceeds max_interval {interval_limit!r}"
                )
        previous_time = timestamp

    # Accumulated increment: rotation dR, velocity dv, position dp, all
    # relative to the starting body frame, beginning at identity and zero.
    delta_rotation = Pose3.identity()
    delta_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    delta_position: tuple[float, float, float] = (0.0, 0.0, 0.0)

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

        # Rotate the corrected specific force into the starting body frame
        # using the incremental rotation at the beginning of the interval.
        rotated_accel = delta_rotation.transform_point(corrected_accel)

        delta_position = (
            delta_position[0] + delta_velocity[0] * dt + 0.5 * rotated_accel[0] * dt * dt,
            delta_position[1] + delta_velocity[1] * dt + 0.5 * rotated_accel[1] * dt * dt,
            delta_position[2] + delta_velocity[2] * dt + 0.5 * rotated_accel[2] * dt * dt,
        )
        delta_velocity = (
            delta_velocity[0] + rotated_accel[0] * dt,
            delta_velocity[1] + rotated_accel[1] * dt,
            delta_velocity[2] + rotated_accel[2] * dt,
        )
        _finite_vector3(delta_position, "delta position")
        _finite_vector3(delta_velocity, "delta velocity")

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
        delta_rotation = delta_rotation.compose(rotation_step)
        _finite_vector3(delta_rotation.quaternion[1:], "delta rotation")
        if not math.isfinite(delta_rotation.quaternion[0]):
            raise ValueError("pre-integration produced non-finite delta rotation")

    duration = records[-1][0] - records[0][0]
    if not math.isfinite(duration):
        raise ValueError("pre-integration produced non-finite duration")
    delta_pose = Pose3(delta_position, delta_rotation.quaternion)
    return PreintegratedImu(
        delta_pose=delta_pose,
        delta_velocity=delta_velocity,
        duration=duration,
        intervals=len(records) - 1,
    )
