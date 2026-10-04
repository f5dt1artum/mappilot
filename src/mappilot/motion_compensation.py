"""Motion compensation (de-skewing) for one timestamped laser scan.

A spinning or scanning LiDAR integrates a frame over time: each beam is
measured at a slightly different instant, so points taken from a moving
platform are skewed in space. :func:`deskew_point_cloud` undoes that skew by
re-expressing every point as if all beams had fired at one reference instant.

The inputs follow the :mod:`mappilot.geometry` convention that a pose maps
local coordinates to parent coordinates
(``p_parent = R * p_local + t``):

* ``trajectory`` samples are ``(time, Pose3)`` pairs; each :class:`Pose3`
  maps body coordinates to world coordinates at an absolute timestamp.
* ``sensor_to_body`` maps sensor coordinates to body coordinates.
* Each scan record holds the beam time as a seconds offset relative to
  ``scan_time`` and one 3D point in sensor coordinates.

For a beam fired at ``tau`` the body pose is interpolated from the two
trajectory samples bracketing ``tau``: translation componentwise linear,
rotation along the unit-quaternion shortest arc. A point whose time lands
exactly on a sample uses that sample directly. The point is transformed to
the world frame through the interpolated body pose and ``sensor_to_body``,
then pulled into the sensor frame at ``reference_time`` through the inverse
of that frame's sensor-to-world pose:

``p_ref = T_sensor_world(ref)^{-1} * T_body_world(tau)
          * T_sensor_body * p``.

This module is pure Python and depends only on the standard library plus
:mod:`mappilot.geometry`. Importing it never starts the HTTP service.

Validation policy mirrors :mod:`mappilot.geometry`: non-real scalars
(including booleans), non-iterable inputs, malformed records, wrong point
dimensions, and trajectory poses or extrinsics that are not :class:`Pose3`
raise :class:`TypeError`; empty point clouds, trajectories with fewer than
two samples, duplicate or out-of-order timestamps, non-finite values, points
or ``reference_time`` outside the trajectory's closed time interval, and an
interpolation span longer than ``max_interpolation_gap`` raise
:class:`ValueError`.
"""

from __future__ import annotations

import math
import numbers
from typing import Iterable

from .geometry import Pose3

__all__ = ["deskew_point_cloud"]


def _real_scalar(value: object, name: str) -> float:
    """Coerce one real, finite, non-boolean scalar to float."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {result!r}")
    return result


def _point(value: object, name: str) -> tuple[float, float, float]:
    """Validate one 3D point of finite, non-boolean real coordinates."""
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be a sequence of 3 real numbers")
    try:
        items = tuple(value)  # type: ignore[call-overload]
    except TypeError:
        raise TypeError(f"{name} must be a sequence of 3 real numbers") from None
    if len(items) != 3:
        raise TypeError(f"{name} must have 3 coordinates, got {len(items)}")
    return (
        _real_scalar(items[0], f"{name}[0]"),
        _real_scalar(items[1], f"{name}[1]"),
        _real_scalar(items[2], f"{name}[2]"),
    )


def _unpack_record(record: object, index: int) -> tuple[float, tuple[float, float, float]]:
    """Validate one ``(time_offset, point)`` scan record."""
    if isinstance(record, (str, bytes, bytearray)):
        raise TypeError(f"scan_points[{index}] must be a (time_offset, point) pair")
    try:
        items = tuple(record)  # type: ignore[call-overload]
    except TypeError:
        raise TypeError(
            f"scan_points[{index}] must be a (time_offset, point) pair"
        ) from None
    if len(items) != 2:
        raise TypeError(
            f"scan_points[{index}] must have exactly 2 elements "
            f"(time_offset, point), got {len(items)}"
        )
    offset = _real_scalar(items[0], f"scan_points[{index}][0]")
    point = _point(items[1], f"scan_points[{index}][1]")
    return offset, point


def _slerp(
    q0: tuple[float, float, float, float],
    q1: tuple[float, float, float, float],
    alpha: float,
) -> tuple[float, float, float, float]:
    """Interpolate two unit quaternions along the shortest arc.

    ``alpha`` is the normalized position in ``[0, 1]``. The returned
    quaternion is canonicalized through :class:`Pose3` construction by the
    caller, so the sign convention here only needs to be self-consistent.
    """
    dot = q0[0] * q1[0] + q0[1] * q1[1] + q0[2] * q1[2] + q0[3] * q1[3]
    # Take the shorter of the two arcs q0 -> q1 and q0 -> -q1.
    if dot < 0.0:
        q1 = (-q1[0], -q1[1], -q1[2], -q1[3])
        dot = -dot
    if dot > 0.9999995:
        # Nearly identical orientations: linear interpolation avoids the
        # division by a vanishing sin(angle).
        blended = (
            q0[0] + alpha * (q1[0] - q0[0]),
            q0[1] + alpha * (q1[1] - q0[1]),
            q0[2] + alpha * (q1[2] - q0[2]),
            q0[3] + alpha * (q1[3] - q0[3]),
        )
        norm = math.sqrt(sum(component * component for component in blended))
        return tuple(component / norm for component in blended)  # type: ignore[return-value]
    angle = math.acos(max(-1.0, min(1.0, dot)))
    sin_angle = math.sin(angle)
    w0 = math.sin((1.0 - alpha) * angle) / sin_angle
    w1 = math.sin(alpha * angle) / sin_angle
    return (
        w0 * q0[0] + w1 * q1[0],
        w0 * q0[1] + w1 * q1[1],
        w0 * q0[2] + w1 * q1[2],
        w0 * q0[3] + w1 * q1[3],
    )


def _interpolate_pose(
    time: float,
    times: list[float],
    poses: list[Pose3],
    max_gap: float | None,
) -> Pose3:
    """Body-to-world pose at ``time`` by interpolation between samples.

    Times are strictly increasing and ``time`` is within their closed range.
    An exact sample hit returns that :class:`Pose3` directly, regardless of
    ``max_gap``; otherwise the bracketing interval must not exceed
    ``max_gap``.
    """
    # Binary search for the right bracketing interval (bisect from the
    # standard library would do, but keeping the index logic local mirrors
    # the rest of the codebase's dependency-free style).
    lo, hi = 0, len(times) - 1
    if time <= times[0]:
        return poses[0]
    if time >= times[-1]:
        return poses[-1]
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if times[mid] <= time:
            lo = mid
        else:
            hi = mid

    t0, t1 = times[lo], times[hi]
    if time == t0:
        return poses[lo]
    if time == t1:
        return poses[hi]
    span = t1 - t0
    if max_gap is not None and span > max_gap:
        raise ValueError(
            f"trajectory gap between samples {t0!r} and {t1!r} is {span!r}, "
            f"exceeds max_interpolation_gap {max_gap!r}"
        )
    alpha = (time - t0) / span
    p0 = poses[lo]
    p1 = poses[hi]
    translation0 = p0.translation
    translation1 = p1.translation
    translation = (
        translation0[0] + alpha * (translation1[0] - translation0[0]),
        translation0[1] + alpha * (translation1[1] - translation0[1]),
        translation0[2] + alpha * (translation1[2] - translation0[2]),
    )
    quaternion = _slerp(p0.quaternion, p1.quaternion, alpha)
    return Pose3(translation, quaternion)


def deskew_point_cloud(
    scan_points: Iterable[tuple[float, tuple[float, float, float]]],
    scan_time: float,
    trajectory: Iterable[tuple[float, Pose3]],
    sensor_to_body: Pose3,
    reference_time: float | None = None,
    max_interpolation_gap: float | None = None,
) -> tuple[tuple[float, float, float], ...]:
    """Compensate one scan's per-point motion distortion to a reference time.

    Parameters
    ----------
    scan_points:
        Scan records in their original order; each record is a pair
        ``(time_offset, point)`` where ``time_offset`` is the beam time in
        seconds relative to ``scan_time`` and ``point`` is a finite 3D point
        in sensor coordinates. The iterable is consumed exactly once; an
        empty scan yields an empty tuple.
    scan_time:
        Absolute timestamp (seconds) of the scan origin; beam absolute times
        are ``scan_time + time_offset``.
    trajectory:
        ``(time, pose)`` pairs with strictly increasing absolute timestamps;
        each pose maps body coordinates to world coordinates. At least two
        samples are required.
    sensor_to_body:
        :class:`Pose3` mapping sensor coordinates to body coordinates.
    reference_time:
        Absolute timestamp (seconds) to which every point is compensated.
        Defaults to ``scan_time``.
    max_interpolation_gap:
        ``None`` (default) or a positive finite number: when the two
        trajectory samples bracketing an interpolated time are farther apart
        than this, :class:`ValueError` is raised. Times landing exactly on a
        sample are exempt.

    Returns
    -------
    tuple of 3-tuples
        Compensated points as immutable ``(x, y, z)`` tuples in the sensor
        frame at ``reference_time``, in the same order and count as the input.

    Raises
    ------
    TypeError
        Non-iterable inputs, malformed records, wrong point dimensions,
        non-real (or boolean) times/coordinates, or trajectory poses /
        extrinsics that are not :class:`Pose3`.
    ValueError
        Empty point cloud; fewer than two trajectory samples; duplicate or
        out-of-order timestamps; non-finite values; beam or reference times
        outside the trajectory's closed interval; non-positive
        ``max_interpolation_gap``; or an interpolation gap exceeding it.
    """
    scan_time_value = _real_scalar(scan_time, "scan_time")

    if not isinstance(sensor_to_body, Pose3):
        raise TypeError(
            "sensor_to_body must be a Pose3, got "
            f"{type(sensor_to_body).__name__}"
        )

    if reference_time is None:
        reference_value = scan_time_value
    else:
        reference_value = _real_scalar(reference_time, "reference_time")

    gap = max_interpolation_gap
    if gap is not None:
        gap_value: float | None = _real_scalar(
            gap, "max_interpolation_gap"
        )
        if gap_value <= 0.0:
            raise ValueError(
                "max_interpolation_gap must be positive when given, got "
                f"{gap_value!r}"
            )
    else:
        gap_value = None

    # Materialize the scan records exactly once; tuples copy scalars and the
    # validated point tuples are freshly built, so caller objects are never
    # mutated or retained by reference.
    if isinstance(scan_points, (str, bytes, bytearray)):
        raise TypeError("scan_points must be an iterable of (time_offset, point) pairs")
    try:
        raw_records = list(scan_points)  # type: ignore[call-overload]
    except TypeError:
        raise TypeError(
            "scan_points must be an iterable of (time_offset, point) pairs"
        ) from None
    if not raw_records:
        raise ValueError("scan_points must not be empty")
    records: list[tuple[float, tuple[float, float, float]]] = [
        _unpack_record(record, index) for index, record in enumerate(raw_records)
    ]

    # Materialize and validate the trajectory up front: types, finiteness,
    # strictly increasing timestamps, minimum sample count.
    if isinstance(trajectory, (str, bytes, bytearray)):
        raise TypeError("trajectory must be an iterable of (time, Pose3) pairs")
    try:
        raw_samples = list(trajectory)  # type: ignore[call-overload]
    except TypeError:
        raise TypeError(
            "trajectory must be an iterable of (time, Pose3) pairs"
        ) from None
    if len(raw_samples) < 2:
        raise ValueError(
            "trajectory must contain at least two samples, got "
            f"{len(raw_samples)}"
        )

    times: list[float] = []
    poses: list[Pose3] = []
    previous_time: float | None = None
    for index, sample in enumerate(raw_samples):
        if isinstance(sample, (str, bytes, bytearray)):
            raise TypeError(f"trajectory[{index}] must be a (time, Pose3) pair")
        try:
            items = tuple(sample)  # type: ignore[call-overload]
        except TypeError:
            raise TypeError(
                f"trajectory[{index}] must be a (time, Pose3) pair"
            ) from None
        if len(items) != 2:
            raise TypeError(
                f"trajectory[{index}] must have exactly 2 elements "
                f"(time, pose), got {len(items)}"
            )
        time_value = _real_scalar(items[0], f"trajectory[{index}][0]")
        if previous_time is not None and time_value <= previous_time:
            raise ValueError(
                "trajectory timestamps must be strictly increasing, got "
                f"{previous_time!r} followed by {time_value!r}"
            )
        pose = items[1]
        if not isinstance(pose, Pose3):
            raise TypeError(
                f"trajectory[{index}][1] must be a Pose3, got "
                f"{type(pose).__name__}"
            )
        times.append(time_value)
        poses.append(pose)
        previous_time = time_value

    time_start = times[0]
    time_end = times[-1]

    def _within_interval(value: float, name: str) -> None:
        if value < time_start or value > time_end:
            raise ValueError(
                f"{name} {value!r} is outside the trajectory interval "
                f"[{time_start!r}, {time_end!r}]"
            )

    _within_interval(reference_value, "reference_time")
    beam_times: list[float] = []
    for index, (offset, _) in enumerate(records):
        absolute_time = scan_time_value + offset
        _within_interval(absolute_time, f"scan_points[{index}] absolute time")
        beam_times.append(absolute_time)

    # Sensor-to-world pose at the reference instant.
    reference_body_pose = _interpolate_pose(
        reference_value, times, poses, gap_value
    )
    reference_sensor_pose = reference_body_pose.compose(sensor_to_body)
    reference_inverse = reference_sensor_pose.inverse()

    compensated: list[tuple[float, float, float]] = []
    for (_, point), absolute_time in zip(records, beam_times):
        body_pose = _interpolate_pose(absolute_time, times, poses, gap_value)
        # sensor frame -> body frame -> world frame ...
        world_point = body_pose.compose(sensor_to_body).transform_point(point)
        # ... -> sensor frame at the reference time.
        compensated.append(reference_inverse.transform_point(world_point))
    return tuple(compensated)
