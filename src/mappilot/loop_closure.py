"""Loop-closure detection and geometric verification between keyframes.

The :func:`detect_loop_closures` entry point takes a time-ordered sequence of
keyframes — each a unique integer id, a world-frame
:class:`~mappilot.geometry.Pose3` initial value, and a 3D point cloud in the
frame's local coordinates — and returns the verified loop closures between
sufficiently separated frames.

For every pair of frames whose input positions differ by at least
``min_separation`` the earlier frame is treated as the source and the later
frame as the target:

1. Pre-filter: pairs whose pose origins are farther apart than
   ``translation_threshold``, or whose relative rotation has a minimal
   geodesic angle larger than ``rotation_threshold``, never reach geometric
   verification.
2. The two pose initial values give the scan-matching initial transform
   ``T_late^{-1} * T_early``, which is refined by
   :func:`~mappilot.registration.point_to_point_icp` under its existing
   semantics (same ``max_iterations``, ``tolerance``, and
   ``max_correspondence_distance``).
3. A candidate is accepted only when ICP converges, the final correspondence
   count is at least ``min_correspondences``, and the final RMSE is at most
   ``max_rmse``. Too few valid correspondences, degenerate (e.g. collinear)
   geometry, or exhausted iterations simply reject that one candidate.

The returned ``relative_pose`` follows the pose-graph measurement convention
``T_early.inverse().compose(T_late)`` and equals the inverse of the ICP
scan transform from the early cloud onto the late cloud, so it can be used
directly as the ``measurement`` of an ``(early_id, late_id)`` edge in
:func:`~mappilot.pose_graph.optimize_pose_graph`.

This module is pure Python and depends only on the standard library plus
:mod:`mappilot.geometry` and :mod:`mappilot.registration`. Importing it never
starts the HTTP service.

Validation policy mirrors :mod:`mappilot.registration`: non-iterable
keyframes or clouds, malformed records, non-integer (including boolean) ids,
non-:class:`Pose3` poses, wrong point dimensions, non-real (including
boolean) coordinates, and parameters of the wrong type raise
:class:`TypeError`; duplicate ids, clouds below three points, non-finite
coordinates, a non-positive ``min_separation``, a ``min_correspondences``
below three, negative or non-finite ``translation_threshold``/``max_rmse``,
a ``rotation_threshold`` outside ``[0, pi]``, and the
:func:`~mappilot.registration.point_to_point_icp` parameter ranges raise
:class:`ValueError`.
"""

from __future__ import annotations

import math
import numbers
from typing import Iterable, NamedTuple

from .geometry import Pose3
from .registration import (
    _point_cloud,
    _positive_float,
    _positive_int,
    point_to_point_icp,
)

__all__ = ["LoopClosure", "detect_loop_closures"]


class LoopClosure(NamedTuple):
    """One verified loop closure between an earlier and a later keyframe.

    Immutable. ``relative_pose`` is the pose-graph measurement for the edge
    ``(early_id, late_id)``, i.e. the inverse of the ICP scan transform from
    the early cloud onto the late cloud. ``correspondences`` holds
    ``(early_point_index, late_point_index)`` pairs ordered by ascending
    early index, and ``rmse`` is the root mean squared distance of exactly
    those pairs, so the two always agree.
    """

    early_id: int
    late_id: int
    relative_pose: Pose3
    rmse: float
    correspondences: tuple[tuple[int, int], ...]


# -- validation ---------------------------------------------------------------


def _nonnegative_float(value: object, name: str) -> float:
    """Coerce one real, finite, non-boolean, non-negative scalar to float."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {result!r}")
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative, got {result}")
    return result


def _rotation_angle(value: object, name: str) -> float:
    """Coerce one real, finite, non-boolean angle in ``[0, pi]`` to float."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {result!r}")
    if result < 0.0 or result > math.pi:
        raise ValueError(f"{name} must lie in [0, pi], got {result}")
    return result


def _min_correspondences(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    result = int(value)
    if result < 3:
        raise ValueError(f"{name} must be at least 3, got {result}")
    return result


def _parse_keyframes(
    value: object,
) -> tuple[tuple[int, Pose3, tuple[tuple[float, float, float], ...]], ...]:
    """Validate the ordered keyframe sequence into ``(id, pose, cloud)`` triples."""
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError("keyframes must be an iterable of (id, pose, points) records")
    try:
        raw_frames = tuple(value)  # type: ignore[call-overload]
    except TypeError:
        raise TypeError(
            "keyframes must be an iterable of (id, pose, points) records"
        ) from None

    frames: list[tuple[int, Pose3, tuple[tuple[float, float, float], ...]]] = []
    seen_ids: set[int] = set()
    for position, raw_frame in enumerate(raw_frames):
        name = f"keyframes[{position}]"
        if isinstance(raw_frame, (str, bytes, bytearray)):
            raise TypeError(f"{name} must be a sequence of 3 elements")
        try:
            items = tuple(raw_frame)  # type: ignore[call-overload]
        except TypeError:
            raise TypeError(f"{name} must be a sequence of 3 elements") from None
        if len(items) != 3:
            raise TypeError(f"{name} must have exactly 3 elements, got {len(items)}")

        frame_id = items[0]
        if isinstance(frame_id, bool) or not isinstance(frame_id, numbers.Integral):
            raise TypeError(
                f"{name}[0] must be an integer keyframe id, "
                f"got {type(frame_id).__name__}"
            )
        frame_id = int(frame_id)

        pose = items[1]
        if not isinstance(pose, Pose3):
            raise TypeError(f"{name}[1] must be a Pose3, got {type(pose).__name__}")

        cloud = _point_cloud(items[2], f"{name}[2]")

        if frame_id in seen_ids:
            raise ValueError(f"duplicate keyframe id: {frame_id}")
        seen_ids.add(frame_id)
        frames.append((frame_id, pose, cloud))
    return tuple(frames)


# -- public entry point -------------------------------------------------------


def detect_loop_closures(
    keyframes: Iterable[tuple[int, Pose3, Iterable[Iterable[float]]]],
    min_separation: int = 10,
    translation_threshold: float = 10.0,
    rotation_threshold: float = math.pi / 2.0,
    min_correspondences: int = 3,
    max_rmse: float = 0.5,
    max_iterations: int = 50,
    tolerance: float = 1e-6,
    max_correspondence_distance: float | None = None,
) -> tuple[LoopClosure, ...]:
    """Detect and verify loop closures between time-separated keyframes.

    Parameters
    ----------
    keyframes:
        Time-ordered iterable of ``(id, pose, points)`` records. ``id`` is a
        unique non-boolean integer, ``pose`` the world-frame
        :class:`~mappilot.geometry.Pose3` initial value, and ``points`` an
        iterable of at least three finite ``(x, y, z)`` points in the frame's
        local coordinates. Generators are consumed exactly once and the input
        objects are never modified.
    min_separation:
        Minimum difference between the input positions of two frames for the
        pair to be considered; a positive integer, default 10.
    translation_threshold:
        Pairs whose pose origins are farther apart than this are skipped
        before geometric verification; a non-negative finite number, default
        10.0.
    rotation_threshold:
        Pairs whose relative rotation has a minimal geodesic angle larger
        than this are skipped before geometric verification; a finite number
        in ``[0, pi]``, default ``pi / 2``.
    min_correspondences:
        Minimum number of final ICP correspondences required to accept a
        candidate; an integer of at least 3, default 3.
    max_rmse:
        Maximum final ICP RMSE accepted for a loop closure; a non-negative
        finite number, default 0.5.
    max_iterations, tolerance, max_correspondence_distance:
        Passed through to :func:`~mappilot.registration.point_to_point_icp`
        with exactly its semantics and validation.

    Returns
    -------
    tuple of LoopClosure
        Ordered by ascending input position of the early frame, then of the
        late frame. Empty when no candidate survives. Each element is
        immutable; ``relative_pose`` follows the
        ``T_early.inverse().compose(T_late)`` pose-graph measurement
        convention and can be used directly as an edge measurement in
        :func:`~mappilot.pose_graph.optimize_pose_graph`.
    """
    frames = _parse_keyframes(keyframes)
    separation = _positive_int(min_separation, "min_separation")
    translation_limit = _nonnegative_float(translation_threshold, "translation_threshold")
    rotation_limit = _rotation_angle(rotation_threshold, "rotation_threshold")
    pair_minimum = _min_correspondences(min_correspondences, "min_correspondences")
    rmse_limit = _nonnegative_float(max_rmse, "max_rmse")
    iteration_limit = _positive_int(max_iterations, "max_iterations")
    convergence_tolerance = _positive_float(tolerance, "tolerance")
    if max_correspondence_distance is not None:
        _positive_float(max_correspondence_distance, "max_correspondence_distance")

    closures: list[LoopClosure] = []
    count = len(frames)
    for early_index in range(count):
        early_id, early_pose, early_points = frames[early_index]
        for late_index in range(early_index + separation, count):
            late_id, late_pose, late_points = frames[late_index]

            offset = math.dist(early_pose.translation, late_pose.translation)
            if offset > translation_limit:
                continue
            relative = early_pose.inverse().compose(late_pose)
            omega = relative.log()
            if math.hypot(omega[0], omega[1], omega[2]) > rotation_limit:
                continue

            initial = late_pose.inverse().compose(early_pose)
            try:
                result = point_to_point_icp(
                    early_points,
                    late_points,
                    initial_pose=initial,
                    max_iterations=iteration_limit,
                    tolerance=convergence_tolerance,
                    max_correspondence_distance=max_correspondence_distance,
                )
            except ValueError:
                # Too few valid correspondences or degenerate geometry only
                # rejects this one candidate.
                continue
            if not result.converged:
                continue
            if len(result.correspondences) < pair_minimum:
                continue
            if result.rmse > rmse_limit:
                continue
            closures.append(
                LoopClosure(
                    early_id,
                    late_id,
                    result.pose.inverse(),
                    result.rmse,
                    result.correspondences,
                )
            )
    return tuple(closures)
