"""Trajectory alignment and accuracy evaluation against a reference trajectory.

The :func:`evaluate_trajectory` entry point consumes two time-ordered
trajectories of ``(id, Pose3)`` records, pairs the records that share an
integer id (in reference trajectory order, unmatched records ignored), and
produces an immutable accuracy report:

* ``alignment="none"`` leaves the estimated poses untouched (identity
  ``alignment_pose``, ``scale`` 1).
* ``alignment="se3"`` (default) fits the rigid transform minimizing the sum
  of squared distances between matched pose origins (SVD of the centered
  cross-covariance, determinant +1 enforced); ``scale`` is 1.
* ``alignment="sim3"`` additionally fits a strictly positive uniform scale.
  The scale multiplies translations only; the alignment rotation is
  left-multiplied onto the estimated rotations.

Per-frame translation error is the Euclidean distance between the aligned
estimated origin and the reference origin; per-frame rotation error is the
minimal geodesic angle of the relative rotation. Relative errors compare,
for every pair of matched frames ``delta`` apart in the matched sequence,
the reference relative pose against the aligned estimated relative pose:
translation is the norm of the error transform's translation, rotation its
minimal geodesic angle. Every reported RMSE is the root mean square of the
corresponding public error sequence, so the two always agree.

This module is pure Python and depends only on the standard library plus
:mod:`mappilot.geometry` and :mod:`mappilot.registration`. Importing it
never starts the HTTP service.

Validation policy mirrors :mod:`mappilot.geometry`: non-iterable
trajectories, records that are not two-element sequences, ids that are not
non-boolean integers, poses that are not :class:`~mappilot.geometry.Pose3`,
a non-string ``alignment``, and a non-integer (or boolean) ``delta`` raise
:class:`TypeError`; duplicate ids within one trajectory, an unsupported
``alignment`` value, ``delta`` below 1, fewer than three common ids, a
``delta`` not smaller than the number of matches, matched positions that
cannot uniquely determine a 3D rotation (under ``se3``/``sim3``), and a
non-positive or non-finite ``sim3`` scale raise :class:`ValueError`.
``alignment="none"`` never fails on degenerate positions.
"""

from __future__ import annotations

import math
import numbers
from typing import Iterable, NamedTuple

from .geometry import Pose3
from .registration import _EPS, _det3, _svd3

__all__ = ["TrajectoryEvaluationResult", "evaluate_trajectory"]

_ALIGNMENTS = ("none", "se3", "sim3")


class TrajectoryEvaluationResult(NamedTuple):
    """Outcome of :func:`evaluate_trajectory`.

    Immutable. ``matched_ids`` holds the common ids in reference trajectory
    order; ``aligned_poses`` and every per-frame error sequence follow the
    same order. ``alignment_pose`` is the fitted alignment (identity for
    ``alignment="none"``) and ``scale`` the fitted uniform scale (1 unless
    ``alignment="sim3"``). Each RMSE is recomputed from the corresponding
    public error sequence, so they always agree.
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


# -- validation ---------------------------------------------------------------


def _read_trajectory(value: object, name: str) -> dict[int, Pose3]:
    """Consume an iterable of ``(id, Pose3)`` records once into an ordered map."""
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be an iterable of (id, Pose3) records")
    try:
        records = list(value)  # type: ignore[call-overload]
    except TypeError:
        raise TypeError(
            f"{name} must be an iterable of (id, Pose3) records"
        ) from None

    poses: dict[int, Pose3] = {}
    for index, record in enumerate(records):
        if isinstance(record, (str, bytes, bytearray)):
            raise TypeError(f"{name}[{index}] must be a (id, Pose3) sequence")
        try:
            pair = tuple(record)
        except TypeError:
            raise TypeError(
                f"{name}[{index}] must be a (id, Pose3) sequence"
            ) from None
        if len(pair) != 2:
            raise TypeError(
                f"{name}[{index}] must have 2 elements, got {len(pair)}"
            )
        record_id, pose = pair
        if isinstance(record_id, bool) or not isinstance(record_id, numbers.Integral):
            raise TypeError(
                f"{name}[{index}] id must be an integer, "
                f"got {type(record_id).__name__}"
            )
        if not isinstance(pose, Pose3):
            raise TypeError(
                f"{name}[{index}] pose must be a Pose3, got {type(pose).__name__}"
            )
        record_id = int(record_id)
        if record_id in poses:
            raise ValueError(f"{name} contains duplicate id {record_id}")
        poses[record_id] = pose
    return poses


# -- alignment solve ------------------------------------------------------------


def _centroid(points: tuple[tuple[float, float, float], ...]) -> tuple[float, float, float]:
    count = len(points)
    return tuple(sum(point[k] for point in points) / count for k in range(3))  # type: ignore[return-value]


def _solve_alignment(
    reference_points: tuple[tuple[float, float, float], ...],
    estimated_points: tuple[tuple[float, float, float], ...],
    with_scale: bool,
) -> tuple[Pose3, float]:
    """Fit the similarity transform ``estimated -> reference`` on matched origins.

    Minimizes ``sum ||r_i - (s * R * e_i + t)||^2`` over the rotation ``R``
    (determinant +1 enforced), the translation ``t``, and -- only when
    ``with_scale`` -- the uniform scale ``s``. Returns the alignment pose
    ``(R, t)`` and the scale (1 when ``with_scale`` is false).
    """
    count = len(reference_points)
    mean_r = _centroid(reference_points)
    mean_e = _centroid(estimated_points)
    centered_r = [
        tuple(point[k] - mean_r[k] for k in range(3)) for point in reference_points
    ]
    centered_e = [
        tuple(point[k] - mean_e[k] for k in range(3)) for point in estimated_points
    ]

    cross = [[0.0, 0.0, 0.0] for _ in range(3)]
    for source, target in zip(centered_e, centered_r):
        for i in range(3):
            row = cross[i]
            source_i = source[i]
            for j in range(3):
                row[j] += source_i * target[j]

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

    if with_scale:
        denominator = sum(sum(c * c for c in source) for source in centered_e)
        numerator = 0.0
        for source, target in zip(centered_e, centered_r):
            for k in range(3):
                rotated = (
                    rotation[k][0] * source[0]
                    + rotation[k][1] * source[1]
                    + rotation[k][2] * source[2]
                )
                numerator += target[k] * rotated
        scale = numerator / denominator
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(
                f"sim3 alignment produced a non-positive or non-finite scale: "
                f"{scale!r}"
            )
    else:
        scale = 1.0

    translation = tuple(
        mean_r[k]
        - scale
        * (
            rotation[k][0] * mean_e[0]
            + rotation[k][1] * mean_e[1]
            + rotation[k][2] * mean_e[2]
        )
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
    """Left-multiply the alignment rotation and scale the aligned translation."""
    rotated = alignment_pose.compose(pose)
    if scale == 1.0:
        return rotated
    alignment_translation = alignment_pose.translation
    rotated_translation = rotated.translation
    translation = tuple(
        alignment_translation[k]
        + scale * (rotated_translation[k] - alignment_translation[k])
        for k in range(3)
    )
    return Pose3(translation, rotated.quaternion)


# -- error metrics --------------------------------------------------------------


def _geodesic_angle(rotation_error: Pose3) -> float:
    """Minimal geodesic angle of a relative rotation, in ``[0, pi]``."""
    w, x, y, z = rotation_error.quaternion
    # Pose3 canonicalizes w >= 0, so this is already the minimal angle.
    return 2.0 * math.atan2(math.hypot(x, y, z), w)


def _rmse(errors: tuple[float, ...]) -> float:
    return math.sqrt(sum(error * error for error in errors) / len(errors))


# -- public entry point ---------------------------------------------------------


def evaluate_trajectory(
    reference_trajectory: Iterable[tuple[int, Pose3]],
    estimated_trajectory: Iterable[tuple[int, Pose3]],
    alignment: str = "se3",
    delta: int = 1,
) -> TrajectoryEvaluationResult:
    """Evaluate an estimated trajectory against a reference trajectory.

    Parameters
    ----------
    reference_trajectory, estimated_trajectory:
        Time-ordered iterables of ``(id, Pose3)`` records. Ids must be
        non-boolean integers, unique within each trajectory. Each iterable is
        consumed exactly once and the inputs are never modified.
    alignment:
        ``"none"`` keeps the estimated poses unchanged; ``"se3"`` (default)
        fits the rigid transform minimizing the sum of squared distances
        between matched pose origins; ``"sim3"`` additionally fits a strictly
        positive uniform scale that multiplies translations only.
    delta:
        Positive integer frame interval within the matched sequence used for
        the relative errors; must be smaller than the number of matches.
        Default 1.

    Returns
    -------
    TrajectoryEvaluationResult
        Immutable report. Records sharing an id are paired in reference
        trajectory order; unmatched records are ignored. Requires at least
        three common ids. Under ``se3``/``sim3`` the matched positions must
        uniquely determine a 3D rotation; under ``sim3`` the fitted scale
        must be positive and finite.
    """
    if not isinstance(alignment, str):
        raise TypeError(f"alignment must be a string, got {type(alignment).__name__}")
    if alignment not in _ALIGNMENTS:
        raise ValueError(
            f"alignment must be one of {_ALIGNMENTS}, got {alignment!r}"
        )
    if isinstance(delta, bool) or not isinstance(delta, numbers.Integral):
        raise TypeError(f"delta must be an integer, got {type(delta).__name__}")
    delta = int(delta)
    if delta < 1:
        raise ValueError(f"delta must be a positive integer, got {delta}")

    reference = _read_trajectory(reference_trajectory, "reference_trajectory")
    estimated = _read_trajectory(estimated_trajectory, "estimated_trajectory")

    matched_ids = tuple(record_id for record_id in reference if record_id in estimated)
    count = len(matched_ids)
    if count < 3:
        raise ValueError(
            f"trajectories must share at least three common ids, got {count}"
        )
    if delta >= count:
        raise ValueError(
            f"delta must be smaller than the number of matches, got {delta} "
            f"for {count} matches"
        )

    reference_poses = tuple(reference[record_id] for record_id in matched_ids)
    estimated_poses = tuple(estimated[record_id] for record_id in matched_ids)

    if alignment == "none":
        alignment_pose = Pose3.identity()
        scale = 1.0
        aligned_poses = estimated_poses
    else:
        alignment_pose, scale = _solve_alignment(
            tuple(pose.translation for pose in reference_poses),
            tuple(pose.translation for pose in estimated_poses),
            alignment == "sim3",
        )
        aligned_poses = tuple(
            _apply_alignment(alignment_pose, scale, pose) for pose in estimated_poses
        )

    translation_errors = tuple(
        math.hypot(
            *(
                aligned.translation[k] - reference_poses[index].translation[k]
                for k in range(3)
            )
        )
        for index, aligned in enumerate(aligned_poses)
    )
    rotation_errors = tuple(
        _geodesic_angle(reference_poses[index].inverse().compose(aligned))
        for index, aligned in enumerate(aligned_poses)
    )

    relative_translation: list[float] = []
    relative_rotation: list[float] = []
    for index in range(count - delta):
        reference_relative = reference_poses[index].inverse().compose(
            reference_poses[index + delta]
        )
        estimated_relative = aligned_poses[index].inverse().compose(
            aligned_poses[index + delta]
        )
        error = reference_relative.inverse().compose(estimated_relative)
        relative_translation.append(math.hypot(*error.translation))
        relative_rotation.append(_geodesic_angle(error))
    relative_translation_errors = tuple(relative_translation)
    relative_rotation_errors = tuple(relative_rotation)

    return TrajectoryEvaluationResult(
        alignment_pose=alignment_pose,
        scale=scale,
        matched_ids=matched_ids,
        aligned_poses=aligned_poses,
        translation_errors=translation_errors,
        rotation_errors=rotation_errors,
        translation_rmse=_rmse(translation_errors),
        rotation_rmse=_rmse(rotation_errors),
        relative_translation_errors=relative_translation_errors,
        relative_rotation_errors=relative_rotation_errors,
        relative_translation_rmse=_rmse(relative_translation_errors),
        relative_rotation_rmse=_rmse(relative_rotation_errors),
    )
