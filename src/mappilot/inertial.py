"""IMU preintegration: compress one segment of inertial samples into a delta.

Visual and LiDAR backends both need the motion between two keyframes as a
single relative increment instead of a raw IMU stream.
:func:`preintegrate_imu` integrates one time-ordered run of IMU samples into
exactly that: a cumulative rotation, velocity, and position delta expressed
in the body frame at the segment start, ready to be composed with a
:class:`~mappilot.geometry.Pose3` or to seed a factor-graph edge.

Conventions follow :mod:`mappilot.geometry` (right-handed coordinates, a pose
maps local coordinates to parent coordinates):

* Each sample is ``(timestamp, acceleration, angular_velocity)`` with the
  timestamp in absolute seconds and the latter two measured in the body
  frame at the sampling instant. ``acceleration`` is the specific force
  (accelerometer output); gravity compensation is the caller's job.
* ``accelerometer_bias`` and ``gyroscope_bias`` are constant over the whole
  segment and subtracted from every measurement.
* The interval ``[t[k], t[k+1]]`` propagates the state with the
  bias-corrected measurement of the sample at ``t[k]`` (left endpoint).

Integration starts from identity rotation, zero velocity, and zero position.
For each interval of length ``dt`` with corrected specific force ``a`` and
corrected angular velocity ``w`` (both in the body frame at ``t[k]``):

1. the specific force is rotated into the segment-start body frame with the
   current delta rotation, ``a_start = dR * a``;
2. position and velocity are updated with
   ``dp' = dp + dv*dt + 0.5*a_start*dt^2`` and ``dv' = dv + a_start*dt``;
3. the delta rotation is right-composed with the rotation of
   ``Pose3.exp((wx*dt, wy*dt, wz*dt, 0, 0, 0))``.

This module is pure Python and depends only on the standard library plus
:mod:`mappilot.geometry`. Importing it never starts the HTTP service.

Validation policy mirrors :mod:`mappilot.geometry`: non-iterable samples or
vectors, malformed records, wrong vector dimensions, and non-real (or
boolean) scalars — including a ``max_interval`` that is neither ``None``
nor a real number — raise :class:`TypeError`; fewer than two samples,
non-finite values, duplicate or out-of-order timestamps, a non-positive or
non-finite ``max_interval``, an interval strictly longer than
``max_interval``, or an accumulation that overflows to a non-finite result
raise :class:`ValueError`.
"""

from __future__ import annotations

import math
import numbers
from typing import Iterable, NamedTuple

from .geometry import Pose3

__all__ = ["ImuPreintegration", "preintegrate_imu"]

_ZERO_VECTOR = (0.0, 0.0, 0.0)


class ImuPreintegration(NamedTuple):
    """Immutable result of preintegrating one IMU segment.

    ``delta_pose`` holds the accumulated position delta in its translation
    and the accumulated rotation delta in its rotation, both expressed in
    the body frame at the segment start. ``delta_velocity`` is the velocity
    delta in that same frame. ``duration`` is the elapsed time
    ``t[-1] - t[0]`` in seconds and ``intervals`` the number of integrated
    intervals (one less than the number of samples).
    """

    delta_pose: Pose3
    delta_velocity: tuple[float, float, float]
    duration: float
    intervals: int


def _real_scalar(value: object, name: str) -> float:
    """Coerce one real, finite, non-boolean scalar to float."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {result!r}")
    return result


def _vector3(value: object, name: str) -> tuple[float, float, float]:
    """Validate one 3D vector of finite, non-boolean real components."""
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be a sequence of 3 real numbers")
    try:
        items = tuple(value)  # type: ignore[call-overload]
    except TypeError:
        raise TypeError(f"{name} must be a sequence of 3 real numbers") from None
    if len(items) != 3:
        raise TypeError(f"{name} must have 3 elements, got {len(items)}")
    return (
        _real_scalar(items[0], f"{name}[0]"),
        _real_scalar(items[1], f"{name}[1]"),
        _real_scalar(items[2], f"{name}[2]"),
    )


def _unpack_sample(
    sample: object, index: int
) -> tuple[float, tuple[float, float, float], tuple[float, float, float]]:
    """Validate one ``(timestamp, acceleration, angular_velocity)`` record."""
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
    timestamp = _real_scalar(items[0], f"samples[{index}][0]")
    acceleration = _vector3(items[1], f"samples[{index}][1]")
    angular_velocity = _vector3(items[2], f"samples[{index}][2]")
    return timestamp, acceleration, angular_velocity


def _require_finite(values: Iterable[float], name: str) -> None:
    """Reject an accumulation step that overflowed to a non-finite value."""
    if not all(math.isfinite(value) for value in values):
        raise ValueError(
            f"preintegration accumulated a non-finite {name}; "
            "the inputs are finite but the segment cannot be represented"
        )


def preintegrate_imu(
    samples: Iterable[
        tuple[float, Iterable[float], Iterable[float]]
    ],
    accelerometer_bias: Iterable[float] = _ZERO_VECTOR,
    gyroscope_bias: Iterable[float] = _ZERO_VECTOR,
    max_interval: float | None = None,
) -> ImuPreintegration:
    """Preintegrate one time-ordered run of IMU samples into a single delta.

    Parameters
    ----------
    samples:
        ``(timestamp, acceleration, angular_velocity)`` records with
        strictly increasing absolute timestamps (seconds) and at least two
        entries. ``acceleration`` is the specific force and
        ``angular_velocity`` the rotation rate, both 3D vectors measured in
        the body frame at the sampling instant. The iterable is consumed
        exactly once and never mutated.
    accelerometer_bias:
        Constant 3D specific-force bias subtracted from every acceleration
        measurement. Defaults to zero.
    gyroscope_bias:
        Constant 3D rotation-rate bias subtracted from every angular
        velocity measurement. Defaults to zero.
    max_interval:
        ``None`` (default) or a positive finite number: any interval
        ``t[k+1] - t[k]`` strictly longer than this raises
        :class:`ValueError`. An interval exactly equal to the limit is
        accepted.

    Returns
    -------
    ImuPreintegration
        Immutable ``(delta_pose, delta_velocity, duration, intervals)``.
        ``delta_pose.translation`` is the position delta and
        ``delta_pose``'s rotation the accumulated rotation delta, both in
        the segment-start body frame; ``delta_velocity`` is the velocity
        delta in that frame. All-zero measurements yield the identity
        ``delta_pose`` and a zero ``delta_velocity``.

    Raises
    ------
    TypeError
        Non-iterable samples or vectors, malformed records, wrong vector
        dimensions, non-real (or boolean) scalars, or a ``max_interval``
        that is neither ``None`` nor a real number.
    ValueError
        Fewer than two samples; non-finite values; duplicate or
        out-of-order timestamps; non-positive or non-finite
        ``max_interval``; an interval strictly exceeding ``max_interval``;
        or finite inputs whose accumulation overflows to a non-finite
        result (no partial result is returned).
    """
    accel_bias = _vector3(accelerometer_bias, "accelerometer_bias")
    gyro_bias = _vector3(gyroscope_bias, "gyroscope_bias")

    if max_interval is None:
        interval_limit: float | None = None
    else:
        interval_limit = _real_scalar(max_interval, "max_interval")
        if interval_limit <= 0.0:
            raise ValueError(
                "max_interval must be positive when given, got "
                f"{interval_limit!r}"
            )

    # Materialize the records exactly once; validated tuples are freshly
    # built, so caller objects are never mutated or retained by reference.
    if isinstance(samples, (str, bytes, bytearray)):
        raise TypeError(
            "samples must be an iterable of "
            "(timestamp, acceleration, angular_velocity) triples"
        )
    try:
        raw_samples = list(samples)  # type: ignore[call-overload]
    except TypeError:
        raise TypeError(
            "samples must be an iterable of "
            "(timestamp, acceleration, angular_velocity) triples"
        ) from None
    if len(raw_samples) < 2:
        raise ValueError(
            f"samples must contain at least two records, got {len(raw_samples)}"
        )
    records = [
        _unpack_sample(sample, index) for index, sample in enumerate(raw_samples)
    ]

    previous_time: float | None = None
    for index, (timestamp, _, _) in enumerate(records):
        if previous_time is not None and timestamp <= previous_time:
            raise ValueError(
                "sample timestamps must be strictly increasing, got "
                f"{previous_time!r} followed by {timestamp!r} at samples[{index}]"
            )
        previous_time = timestamp

    delta_rotation = Pose3.identity()
    delta_velocity = [0.0, 0.0, 0.0]
    delta_position = [0.0, 0.0, 0.0]

    for index in range(len(records) - 1):
        timestamp, acceleration, angular_velocity = records[index]
        dt = records[index + 1][0] - timestamp
        if interval_limit is not None and dt > interval_limit:
            raise ValueError(
                f"interval between samples[{index}] and samples[{index + 1}] "
                f"is {dt!r}, exceeds max_interval {interval_limit!r}"
            )

        corrected_force = (
            acceleration[0] - accel_bias[0],
            acceleration[1] - accel_bias[1],
            acceleration[2] - accel_bias[2],
        )
        corrected_rate = (
            angular_velocity[0] - gyro_bias[0],
            angular_velocity[1] - gyro_bias[1],
            angular_velocity[2] - gyro_bias[2],
        )

        # Rotate the specific force into the segment-start body frame with
        # the delta rotation at the interval start (zero translation, so
        # transform_point is a pure rotation here).
        force_start = delta_rotation.transform_point(corrected_force)
        for axis in range(3):
            delta_position[axis] += (
                delta_velocity[axis] * dt + 0.5 * force_start[axis] * dt * dt
            )
            delta_velocity[axis] += force_start[axis] * dt

        # Right-compose the rotation increment of this interval.
        delta_rotation = delta_rotation.compose(
            Pose3.exp(
                (
                    corrected_rate[0] * dt,
                    corrected_rate[1] * dt,
                    corrected_rate[2] * dt,
                    0.0,
                    0.0,
                    0.0,
                )
            )
        )

        _require_finite(delta_position, "position delta")
        _require_finite(delta_velocity, "velocity delta")
        _require_finite(delta_rotation.quaternion, "rotation delta")

    duration = records[-1][0] - records[0][0]
    _require_finite((duration,), "duration")

    return ImuPreintegration(
        delta_pose=Pose3(tuple(delta_position), delta_rotation.quaternion),
        delta_velocity=tuple(delta_velocity),
        duration=duration,
        intervals=len(records) - 1,
    )
