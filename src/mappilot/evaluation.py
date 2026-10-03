"""Trajectory alignment and accuracy evaluation against a reference trajectory.

The :func:`evaluate_trajectory` entry point consumes two trajectories of
``(id, Pose3)`` records in time order, pairs them by shared integer ids (in
reference trajectory order, unmatched records ignored), optionally aligns the
estimated trajectory onto the reference, and reports absolute and relative
pose errors. Three alignment modes are supported:

* ``"none"``: the estimated poses are evaluated as-is; the reported
  ``alignment_pose`` is the identity and ``scale`` is 1.
* ``"se3"`` (default): the rigid transform minimizing the sum of squared
  distances between all matched positions is fitted (SVD of the centered
  cross-covariance, determinant +1 enforced) and left-composed onto every
  estimated pose; ``scale`` is 1.
* ``"sim3"``: additionally fits a strictly positive uniform scale. The scale
  acts on translations only; the alignment rotation is still left-composed
  onto the estimated rotations unscaled.

Per-frame translation errors are the Euclidean distances between aligned and
reference origins; per-frame rotation errors are the minimal geodesic angles
of the relative rotations. Relative errors compare, for every pair of matched
frames separated by ``delta`` steps, the reference relative pose against the
aligned estimated relative pose: the translation part is the norm of the
error transform's translation, the rotation part its minimal geodesic angle.
Every reported RMSE is computed from the corresponding public error sequence.

This module is pure Python and depends only on the standard library plus
:mod:`mappilot.geometry` and :mod:`mappilot.registration`. Importing it never
starts the HTTP service.

Validation policy mirrors :mod:`mappilot.geometry` and
:mod:`mappilot.registration`: non-iterable trajectories, records that are not
``(id, pose)`` pairs, non-integer (or boolean) ids, non-:class:`Pose3` poses,
a non-string ``alignment``, and a non-integer (or boolean) ``delta`` raise
:class:`TypeError`; duplicate ids within either trajectory, unsupported
``alignment`` values, ``delta`` below 1, fewer than three common ids, a
``delta`` not smaller than the number of matches, matched positions that
cannot uniquely determine a 3D rotation (``"se3"``/``"sim3"`` only), and a
non-positive or non-finite fitted scale (``"sim3"`` only) raise
:class:`ValueError`. ``"none"`` never fails on degenerate positions.
"""

from __future__ import annotations

import math
import numbers
from typing import Iterable, NamedTuple

from .geometry import Pose3
from .registration import _EPS, _det3, _svd3

__all__ = ["TrajectoryEvaluation", "evaluate_trajectory"]

_ALIGNMENTS = ("none", "se3", "sim3")

# Minimum number of matched poses required for a meaningful evaluation.
_MIN_MATCHES = 3


class TrajectoryEvaluation(NamedTuple):
    """Outcome of :func:`evaluate_trajectory`.

    Immutable. ``matched_ids`` holds the shared ids in reference trajectory
    order; ``aligned_poses``, ``translation_errors``, and ``rotation_errors``
    follow that same order. ``scale`` is 1 unless ``alignment="sim3"``.
    The relative error sequences hold one entry per pair of matched frames
    ``delta`` steps apart, in sequence order, and every RMSE is the root mean
    square of the corresponding public error sequence.
    """

    alignment_pose: Pose3
    scale: float
    matched_ids: tuple[int, ...]
    aligned_poses: tuple[Pose3, ...]
    translation_errors: tuple[float, ...]
    rotation_errors: tuple[float, ...]
    translation_rmse: float
    rotation_rmse: float
    relative_translation_errors: tuple[float, ...]
    relative_rotation_errors: tuple[float, ...]
    relative_translation_rmse: float
    relative_rotation_rmse: float


# -- validation -------------------------------------------------------------


def _record(value: object, name: str) -> tuple[int, Pose3]:
    """Validate one ``(id, Pose3)`` record."""
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be a (id, Pose3) pair")
    try:
        items = tuple(value)  # type: ignore[call-overload]
    except TypeError:
        raise TypeError(f"{name} must be a (id, Pose3) pair") from None
    if len(items) != 2:
        raise TypeError(
            f"{name} must be a (id, Pose3) pair, got {len(items)} elements"
        )
    record_id, pose = items
    if isinstance(record_id, bool) or not isinstance(record_id, numbers.Integral):
        raise TypeError(
            f"{name} id must be an integer, got {type(record_id).__name__}"
        )
    if not isinstance(pose, Pose3):
        raise TypeError(f"{name} pose must be a Pose3, got {type(pose).__name__}")
    return int(record_id), pose


def _trajectory(value: object, name: str) -> tuple[tuple[int, ...], dict[int, Pose3]]:
    """Validate a trajectory; return its ids in order and an id-to-pose map.

    The iterable is consumed exactly once and never modified.
    """
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be an iterable of (id, Pose3) records")
    try:
        records = list(value)  # type: ignore[call-overload]
    except TypeError:
        raise TypeError(
            f"{name} must be an iterable of (id, Pose3) records"
        ) from None
    order: list[int] = []
    poses: dict[int, Pose3] = {}
    for index, raw_record in enumerate(records):
        record_id, pose = _record(raw_record, f"{name}[{index}]")
        if record_id in poses:
            raise ValueError(f"{name} contains duplicate id {record_id}")
        poses[record_id] = pose
        order.append(record_id)
    return tuple(order), poses


# -- alignment --------------------------------------------------------------


def _fit_alignment(
    reference_points: tuple[tuple[float, float, float], ...],
    estimated_points: tuple[tuple[float, float, float], ...],
    with_scale: bool,
) -> tuple[Pose3, float]:
    """Fit the SE(3)/Sim(3) transform mapping estimated positions to reference.

    Returns the alignment pose and the uniform scale (1 without ``with_scale``).
    Raises :class:`ValueError` when the matched positions cannot uniquely
    determine a 3D rotation, or when a fitted scale is not strictly positive
    and finite.
    """
    count = len(reference_points)
    mean_r = [0.0, 0.0, 0.0]
    mean_e = [0.0, 0.0, 0.0]
    for reference, estimated in zip(reference_points, estimated_points):
        for k in range(3):
            mean_r[k] += reference[k]
            mean_e[k] += estimated[k]
    mean_r = [value / count for value in mean_r]
    mean_e = [value / count for value in mean_e]

    centered_r = [
        tuple(point[k] - mean_r[k] for k in range(3)) for point in reference_points
    ]
    centered_e = [
        tuple(point[k] - mean_e[k] for k in range(3)) for point in estimated_points
    ]

    cross = [[0.0, 0.0, 0.0] for _ in range(3)]
    for reference, estimated in zip(centered_r, centered_e):
        for i in range(3):
            row = cross[i]
            e_i = estimated[i]
            for j in range(3):
                row[j] += e_i * reference[j]

    u, singular, vt = _svd3(tuple(tuple(row) for row in cross))

    # Rank 1 (collinear or worse) leaves a rotation about the line free; the
    # 3D rotation is not unique. Rank 2 is still fully constrained.
    rank_tolerance = singular[0] * max(3, count) * _EPS
    if singular[1] <= rank_tolerance:
        raise ValueError(
            "matched positions are too degenerate (collinear) to uniquely "
            "determine a 3D rotation"
        )

    # H = U diag(s) V^T minimizes with R = V diag(1, 1, det(V U^T)) U^T; the
    # sign flip rules out a reflection.
    v_matrix = tuple(tuple(vt[col][row] for col in range(3)) for row in range(3))
    determinant = _det3(v_matrix) * _det3(u)
    reflection_correction = (1.0, 1.0, determinant)
    rotation = tuple(
        tuple(
            sum(
                v_matrix[row][k] * reflection_correction[k] * u[col][k]
                for k in range(3)
            )
            for col in range(3)
        )
        for row in range(3)
    )

    scale = 1.0
    if with_scale:
        denominator = 0.0
        numerator = 0.0
        for reference, estimated in zip(centered_r, centered_e):
            denominator += sum(component * component for component in estimated)
            rotated = tuple(
                sum(rotation[k][j] * estimated[j] for j in range(3)) for k in range(3)
            )
            numerator += sum(reference[k] * rotated[k] for k in range(3))
        scale = numerator / denominator if denominator > 0.0 else math.nan
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(
                "similarity alignment produced a non-positive or non-finite "
                f"scale ({scale!r})"
            )

    translation = tuple(
        mean_r[k] - scale * sum(rotation[k][j] * mean_e[j] for j in range(3))
        for k in range(3)
    )
    alignment_pose = Pose3.from_matrix(
        (
            (rotation[0][0], rotation[0][1], rotation[0][2], translation[0]),
            (rotation[1][0], rotation[1][1], rotation[1][2], translation[1]),
            (rotation[2][0], rotation[2][1], rotation[2][2], translation[2]),
            (0.0, 0.0, 0.0, 1.0),
        )
    )
    return alignment_pose, scale


def _apply_alignment(alignment_pose: Pose3, scale: float, pose: Pose3) -> Pose3:
    """Left-compose the alignment onto ``pose``, scaling only its translation."""
    moved = alignment_pose.transform_point(pose.translation)
    alignment_translation = alignment_pose.translation
    translation = tuple(
        scale * moved[k] + (1.0 - scale) * alignment_translation[k] for k in range(3)
    )
    quaternion = alignment_pose.compose(pose).quaternion
    return Pose3(translation, quaternion)


# -- error metrics ------------------------------------------------------------


def _geodesic_angle(error_pose: Pose3) -> float:
    """Minimal geodesic rotation angle of a relative pose, in ``[0, pi]``."""
    wx, wy, wz = error_pose.log()[:3]
    return math.hypot(wx, wy, wz)


def _rmse(values: tuple[float, ...]) -> float:
    return math.sqrt(sum(value * value for value in values) / len(values))


# -- public entry point -------------------------------------------------------


def evaluate_trajectory(
    reference: Iterable[tuple[int, Pose3]],
    estimated: Iterable[tuple[int, Pose3]],
    alignment: str = "se3",
    delta: int = 1,
) -> TrajectoryEvaluation:
    """Evaluate an estimated trajectory against a reference trajectory.

    Parameters
    ----------
    reference, estimated:
        Iterables of ``(id, Pose3)`` records in time order. Ids must be
        non-boolean integers, unique within each trajectory. Records are
        paired by shared id in reference order; unmatched records are
        ignored. Each iterable is consumed exactly once and never modified.
    alignment:
        ``"none"``, ``"se3"`` (default), or ``"sim3"``. ``"none"`` evaluates
        the estimated poses unchanged; ``"se3"`` fits the rigid transform
        minimizing the sum of squared matched-position distances;
        ``"sim3"`` additionally fits a strictly positive uniform scale that
        acts on translations only.
    delta:
        Frame interval within the matched sequence for the relative pose
        errors; a positive integer smaller than the number of matches,
        default 1.

    Returns
    -------
    TrajectoryEvaluation
        Immutable result. Per-frame errors compare each aligned estimated
        pose against its reference pose (Euclidean origin distance and
        minimal geodesic rotation angle). Relative errors compare, for every
        pair of matched frames ``delta`` apart, the reference relative pose
        against the aligned estimated relative pose. All RMSE values are
        computed from the corresponding public error sequences.
    """
    if not isinstance(alignment, str):
        raise TypeError(
            f"alignment must be a string, got {type(alignment).__name__}"
        )
    if alignment not in _ALIGNMENTS:
        raise ValueError(
            f"alignment must be one of {_ALIGNMENTS}, got {alignment!r}"
        )
    if isinstance(delta, bool) or not isinstance(delta, numbers.Integral):
        raise TypeError(f"delta must be an integer, got {type(delta).__name__}")
    frame_delta = int(delta)
    if frame_delta < 1:
        raise ValueError(f"delta must be a positive integer, got {frame_delta}")

    reference_order, reference_poses = _trajectory(reference, "reference")
    _, estimated_poses = _trajectory(estimated, "estimated")

    matched_ids = tuple(
        record_id for record_id in reference_order if record_id in estimated_poses
    )
    if len(matched_ids) < _MIN_MATCHES:
        raise ValueError(
            f"trajectories must share at least {_MIN_MATCHES} ids, "
            f"got {len(matched_ids)}"
        )
    if frame_delta >= len(matched_ids):
        raise ValueError(
            f"delta ({frame_delta}) must be smaller than the number of "
            f"matched poses ({len(matched_ids)})"
        )

    reference_matched = [reference_poses[record_id] for record_id in matched_ids]
    estimated_matched = [estimated_poses[record_id] for record_id in matched_ids]

    if alignment == "none":
        alignment_pose = Pose3.identity()
        scale = 1.0
        aligned = list(estimated_matched)
    else:
        alignment_pose, scale = _fit_alignment(
            tuple(pose.translation for pose in reference_matched),
            tuple(pose.translation for pose in estimated_matched),
            alignment == "sim3",
        )
        aligned = [
            _apply_alignment(alignment_pose, scale, pose)
            for pose in estimated_matched
        ]

    translation_errors = tuple(
        math.dist(aligned_pose.translation, reference_pose.translation)
        for aligned_pose, reference_pose in zip(aligned, reference_matched)
    )
    rotation_errors = tuple(
        _geodesic_angle(reference_pose.inverse().compose(aligned_pose))
        for aligned_pose, reference_pose in zip(aligned, reference_matched)
    )

    relative_translation: list[float] = []
    relative_rotation: list[float] = []
    for index in range(len(matched_ids) - frame_delta):
        reference_relative = reference_matched[index].inverse().compose(
            reference_matched[index + frame_delta]
        )
        estimated_relative = aligned[index].inverse().compose(
            aligned[index + frame_delta]
        )
        error = reference_relative.inverse().compose(estimated_relative)
        relative_translation.append(math.hypot(*error.translation))
        relative_rotation.append(_geodesic_angle(error))
    relative_translation_errors = tuple(relative_translation)
    relative_rotation_errors = tuple(relative_rotation)

    return TrajectoryEvaluation(
        alignment_pose=alignment_pose,
        scale=scale,
        matched_ids=matched_ids,
        aligned_poses=tuple(aligned),
        translation_errors=translation_errors,
        rotation_errors=rotation_errors,
        translation_rmse=_rmse(translation_errors),
        rotation_rmse=_rmse(rotation_errors),
        relative_translation_errors=relative_translation_errors,
        relative_rotation_errors=relative_rotation_errors,
        relative_translation_rmse=_rmse(relative_translation_errors),
        relative_rotation_rmse=_rmse(relative_rotation_errors),
    )
