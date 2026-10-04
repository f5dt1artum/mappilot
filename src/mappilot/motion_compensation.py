"""Motion compensation (deskewing) of a spinning-scan point cloud.

The :func:`deskew_point_cloud` entry point removes motion distortion from
one scan frame: each point was captured at its own absolute time while the
sensor moved, so every point is re-expressed in the sensor frame of a
single reference time.

Conventions:

* A scan record is ``(time_offset, point)`` where ``time_offset`` is the
  offset in seconds relative to ``scan_time`` and ``point`` is a 3D point
  in sensor coordinates. The absolute capture time of a point is
  ``scan_time + time_offset``.
* A trajectory record is ``(timestamp, pose)`` where ``timestamp`` is an
  absolute time in seconds and ``pose`` is a
  :class:`~mappilot.geometry.Pose3` mapping body coordinates to world
  coordinates. Timestamps must be strictly increasing.
* ``sensor_to_body`` is a :class:`~mappilot.geometry.Pose3` mapping sensor
  coordinates to body coordinates.
* ``reference_time`` defaults to ``scan_time``.

For every point, the body pose at its absolute capture time is interpolated
from the two neighbouring trajectory samples: translation is interpolated
component-wise linearly and rotation is interpolated along the shortest arc
between the unit quaternions (slerp). A capture time that exactly hits a
sample uses that sample's pose directly. The point is then transformed to
world coordinates through ``sensor_to_body`` and the body pose at its
capture time, and finally through the inverse of the sensor world pose at
``reference_time``. Results are returned as immutable 3-tuples in exactly
the input order and count. Input iterables (including generators) are
consumed exactly once and caller-owned objects are never mutated.

Every point capture time and ``reference_time`` must lie inside the closed
trajectory time interval. ``max_interpolation_gap`` is ``None`` or a
positive finite real number; when set, any interpolation that would span
two adjacent trajectory samples farther apart than the gap raises
:class:`ValueError` (an exact sample hit is never rejected).

This module is pure Python and depends only on the standard library plus
:mod:`mappilot.geometry`. Importing it never starts the HTTP service.

Validation policy mirrors :mod:`mappilot.geometry` and
:mod:`mappilot.localization`: non-iterable inputs, malformed records, wrong
point dimensions, non-real (including boolean) times or coordinates, and a
trajectory pose or ``sensor_to_body`` that is not a
:class:`~mappilot.geometry.Pose3` raise :class:`TypeError`; an empty scan,
fewer than two trajectory samples, duplicated or decreasing timestamps, any
non-finite value, times outside the trajectory interval, an interpolation
gap above ``max_interpolation_gap``, and a non-positive
``max_interpolation_gap`` raise :class:`ValueError`.
"""

from __future__ import annotations

import bisect
import math
import numbers
from typing import Iterable

from .geometry import Pose3

__all__ = ["deskew_point_cloud"]

# Above this cosine of the half-angle between two unit quaternions, slerp
# degenerates to a normalized linear blend to avoid dividing by sin(0).
_SLERP_LINEAR_THRESHOLD = 0.9995


# -- validation -------------------------------------------------------------


def _real_scalar(value: object, name: str) -> float:
    """Coerce one real, finite, non-boolean scalar to float."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {result!r}")
    return result


def _record(value: object, name: str) -> tuple[object, object]:
    """Coerce one ``(time, payload)`` record, checking only its shape."""
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be a (time, value) record")
    try:
        items = tuple(value)  # type: ignore[call-overload]
    except TypeError:
        raise TypeError(f"{name} must be a (time, value) record") from None
    if len(items) != 2:
        raise TypeError(f"{name} must have 2 elements, got {len(items)}")
    return items[0], items[1]


def _point(value: object, name: str) -> tuple[float, float, float]:
    """Coerce one 3D point of real, finite coordinates."""
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be a sequence of 3 real numbers")
    try:
        coords = tuple(value)  # type: ignore[call-overload]
    except TypeError:
        raise TypeError(f"{name} must be a sequence of 3 real numbers") from None
    if len(coords) != 3:
        raise TypeError(f"{name} must have 3 coordinates, got {len(coords)}")
    return (
        _real_scalar(coords[0], f"{name}[0]"),
        _real_scalar(coords[1], f"{name}[1]"),
        _real_scalar(coords[2], f"{name}[2]"),
    )


def _scan_records(value: object) -> tuple[tuple[float, tuple[float, float, float]], ...]:
    """Validate a non-empty scan of ``(offset, point)`` records, consumed once."""
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError("scan_points must be an iterable of (time_offset, point) records")
    try:
        raw_records = tuple(value)  # type: ignore[call-overload]
    except TypeError:
        raise TypeError(
            "scan_points must be an iterable of (time_offset, point) records"
        ) from None
    if len(raw_records) == 0:
        raise ValueError("scan_points must contain at least one point, got 0")

    records: list[tuple[float, tuple[float, float, float]]] = []
    for index, raw_record in enumerate(raw_records):
        name = f"scan_points[{index}]"
        raw_offset, raw_point = _record(raw_record, name)
        offset = _real_scalar(raw_offset, f"{name}[0]")
        point = _point(raw_point, f"{name}[1]")
        records.append((offset, point))
    return tuple(records)


def _trajectory_records(value: object) -> tuple[tuple[float, ...], tuple[Pose3, ...]]:
    """Validate a trajectory of ``(timestamp, Pose3)`` records, consumed once."""
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError("trajectory must be an iterable of (timestamp, Pose3) records")
    try:
        raw_records = tuple(value)  # type: ignore[call-overload]
    except TypeError:
        raise TypeError(
            "trajectory must be an iterable of (timestamp, Pose3) records"
        ) from None
    if len(raw_records) < 2:
        raise ValueError(
            f"trajectory must contain at least 2 samples, got {len(raw_records)}"
        )

    times: list[float] = []
    poses: list[Pose3] = []
    for index, raw_record in enumerate(raw_records):
        name = f"trajectory[{index}]"
        raw_time, raw_pose = _record(raw_record, name)
        timestamp = _real_scalar(raw_time, f"{name}[0]")
        if not isinstance(raw_pose, Pose3):
            raise TypeError(
                f"{name}[1] must be a Pose3, got {type(raw_pose).__name__}"
            )
        if times and timestamp <= times[-1]:
            raise ValueError(
                f"trajectory timestamps must be strictly increasing, "
                f"got {timestamp} after {times[-1]}"
            )
        times.append(timestamp)
        poses.append(raw_pose)
    return tuple(times), tuple(poses)


def _max_interpolation_gap(value: object) -> float | None:
    if value is None:
        return None
    result = _real_scalar(value, "max_interpolation_gap")
    if result <= 0.0:
        raise ValueError(f"max_interpolation_gap must be greater than zero, got {result}")
    return result


# -- interpolation ------------------------------------------------------------


def _slerp(
    q0: tuple[float, float, float, float],
    q1: tuple[float, float, float, float],
    ratio: float,
) -> tuple[float, float, float, float]:
    """Interpolate unit quaternions along the shortest arc."""
    dot = q0[0] * q1[0] + q0[1] * q1[1] + q0[2] * q1[2] + q0[3] * q1[3]
    if dot < 0.0:
        q1 = (-q1[0], -q1[1], -q1[2], -q1[3])
        dot = -dot
    if dot > _SLERP_LINEAR_THRESHOLD:
        return tuple(a + ratio * (b - a) for a, b in zip(q0, q1))  # type: ignore[return-value]
    omega = math.acos(min(1.0, dot))
    sin_omega = math.sin(omega)
    scale0 = math.sin((1.0 - ratio) * omega) / sin_omega
    scale1 = math.sin(ratio * omega) / sin_omega
    return (
        scale0 * q0[0] + scale1 * q1[0],
        scale0 * q0[1] + scale1 * q1[1],
        scale0 * q0[2] + scale1 * q1[2],
        scale0 * q0[3] + scale1 * q1[3],
    )


def _interpolate_pose(
    times: tuple[float, ...],
    poses: tuple[Pose3, ...],
    time: float,
    max_gap: float | None,
) -> Pose3:
    """Body pose at ``time``; ``time`` must lie in the closed sample interval."""
    index = bisect.bisect_left(times, time)
    if index < len(times) and times[index] == time:
        return poses[index]

    lo_time, hi_time = times[index - 1], times[index]
    if max_gap is not None and hi_time - lo_time > max_gap:
        raise ValueError(
            f"interpolation gap {hi_time - lo_time} exceeds "
            f"max_interpolation_gap {max_gap}"
        )
    ratio = (time - lo_time) / (hi_time - lo_time)
    lo_pose, hi_pose = poses[index - 1], poses[index]

    lo_t, hi_t = lo_pose.translation, hi_pose.translation
    translation = (
        lo_t[0] + ratio * (hi_t[0] - lo_t[0]),
        lo_t[1] + ratio * (hi_t[1] - lo_t[1]),
        lo_t[2] + ratio * (hi_t[2] - lo_t[2]),
    )
    quaternion = _slerp(lo_pose.quaternion, hi_pose.quaternion, ratio)
    return Pose3(translation, quaternion)


# -- public entry point ---------------------------------------------------------


def deskew_point_cloud(
    scan_points: Iterable[tuple[float, Iterable[float]]],
    scan_time: float,
    trajectory: Iterable[tuple[float, Pose3]],
    sensor_to_body: Pose3,
    reference_time: float | None = None,
    max_interpolation_gap: float | None = None,
) -> tuple[tuple[float, float, float], ...]:
    """Compensate one scan frame for sensor motion, returning reference-time points.

    ``scan_points`` is an iterable of ``(time_offset, point)`` records in
    their original acquisition order; each offset is relative to
    ``scan_time`` and each point is a 3D point in sensor coordinates.
    ``trajectory`` is an iterable of ``(timestamp, pose)`` records with
    strictly increasing absolute timestamps and body-to-world
    :class:`~mappilot.geometry.Pose3` poses. ``sensor_to_body`` maps sensor
    coordinates to body coordinates. ``reference_time`` defaults to
    ``scan_time``.

    Every point is transformed to world coordinates with the interpolated
    sensor world pose at its absolute capture time, then back into the
    sensor frame of ``reference_time``. The result is a tuple of immutable
    3-tuples with the same order and count as the input.
    """
    start = _real_scalar(scan_time, "scan_time")
    reference = start if reference_time is None else _real_scalar(reference_time, "reference_time")
    max_gap = _max_interpolation_gap(max_interpolation_gap)
    if not isinstance(sensor_to_body, Pose3):
        raise TypeError(
            f"sensor_to_body must be a Pose3, got {type(sensor_to_body).__name__}"
        )
    times, poses = _trajectory_records(trajectory)
    records = _scan_records(scan_points)

    first_time, last_time = times[0], times[-1]
    absolute_times: list[float] = []
    for index, (offset, _) in enumerate(records):
        absolute = start + offset
        if not first_time <= absolute <= last_time:
            raise ValueError(
                f"scan_points[{index}] time {absolute} lies outside the "
                f"trajectory interval [{first_time}, {last_time}]"
            )
        absolute_times.append(absolute)
    if not first_time <= reference <= last_time:
        raise ValueError(
            f"reference_time {reference} lies outside the trajectory "
            f"interval [{first_time}, {last_time}]"
        )

    reference_body = _interpolate_pose(times, poses, reference, max_gap)
    reference_sensor_inverse = reference_body.compose(sensor_to_body).inverse()

    compensated: list[tuple[float, float, float]] = []
    for absolute, (_, point) in zip(absolute_times, records):
        body_pose = _interpolate_pose(times, poses, absolute, max_gap)
        sensor_world = body_pose.compose(sensor_to_body)
        world_point = sensor_world.transform_point(point)
        compensated.append(reference_sensor_inverse.transform_point(world_point))
    return tuple(compensated)
