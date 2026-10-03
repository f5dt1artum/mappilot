"""Scan matching: point-to-point Iterative Closest Point.

The :func:`point_to_point_icp` entry estimates the SE(3) rigid transform that
aligns a source point cloud onto a target point cloud. Every iteration:

1. the source points are transformed with the current pose estimate;
2. each transformed source point picks its nearest target point (target points
   may be chosen repeatedly; exact distance ties resolve to the smallest target
   index);
3. pairs farther than ``max_correspondence_distance`` are dropped when a gate
   is configured;
4. the rigid pose minimizing the sum of squared pair distances updates the
   estimate (Horn's closed-form unit-quaternion solution; a unit quaternion
   can only describe a proper rotation, so the fitted rotation never
   contains a mirror).

Convergence is declared when the absolute RMSE change between two consecutive
iterations is at most ``tolerance``. Exhausting ``max_iterations`` returns the
last estimate with ``converged`` False; convergence failures never raise.

The module is pure Python, depending only on the standard library and
:mod:`mappilot.geometry`. Importing it never starts the HTTP service.

Validation policy mirrors :mod:`mappilot.geometry`: non-iterable clouds,
points that are not 3 real scalars, and booleans raise :class:`TypeError`;
NaN or infinite coordinates, fewer than three points, non-positive iteration
counts, non-positive/non-finite tolerances, fewer than three surviving pairs,
and degenerate pair configurations that cannot uniquely pin down a 3D rigid
pose raise :class:`ValueError`.
"""

from __future__ import annotations

import math
import numbers
from typing import Iterable, NamedTuple, Sequence

from .geometry import Pose3

__all__ = ["ICPResult", "point_to_point_icp"]

# Singular values of the (3x3) pair cross-covariance smaller than this
# fraction of its largest one make the correspondence geometry collinear,
# which leaves rotation about the line undetermined. Non-collinear pairs
# (including planar triplets) always pin down a unique proper rotation.
_SINGULAR_RANK_TOL = 1e-10


class ICPResult(NamedTuple):
    """Outcome of an ICP run.

    Attributes:
        pose: Estimated transform mapping source coordinates to target
            coordinates.
        converged: True when an RMSE change no larger than ``tolerance`` was
            observed between two iterations; False if the iteration budget
            ran out.
        iterations: Number of pose updates actually performed.
        rmse: Root mean squared Euclidean distance of the final pairs,
            recomputed under the final pose.
        correspondences: Pairs ``(source_index, target_index)`` kept in the
            final association, sorted by ascending source index.
    """

    pose: Pose3
    converged: bool
    iterations: int
    rmse: float
    correspondences: tuple[tuple[int, int], ...]


def _point(value: object, name: str) -> tuple[float, float, float]:
    """Coerce one finite, non-boolean 3D point to a tuple."""
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be a 3D point")
    try:
        items = tuple(value)  # type: ignore[call-overload]
    except TypeError:
        raise TypeError(f"{name} must be a 3D point") from None
    if len(items) != 3:
        raise TypeError(f"{name} must have 3 coordinates, got {len(items)}")
    result = []
    for index, item in enumerate(items):
        if isinstance(item, bool) or not isinstance(item, numbers.Real):
            raise TypeError(
                f"{name}[{index}] must be a real number, got {type(item).__name__}"
            )
        coordinate = float(item)
        if not math.isfinite(coordinate):
            raise ValueError(f"{name}[{index}] must be finite, got {coordinate!r}")
        result.append(coordinate)
    return (result[0], result[1], result[2])


def _point_cloud(value: object, name: str) -> tuple[tuple[float, float, float], ...]:
    """Coerce a cloud of at least three 3D points to a tuple of tuples."""
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be a sequence of 3D points")
    try:
        items = tuple(value)  # type: ignore[call-overload]
    except TypeError:
        raise TypeError(f"{name} must be a sequence of 3D points") from None
    cloud = tuple(_point(point, f"{name}[{index}]") for index, point in enumerate(items))
    if len(cloud) < 3:
        raise ValueError(f"{name} must contain at least 3 points, got {len(cloud)}")
    return cloud


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer, got {result}")
    return result


def _finite_positive_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive number, got {result!r}")
    return result


def _nearest_targets(
    transformed: Sequence[tuple[float, float, float]],
    target: Sequence[tuple[float, float, float]],
    gate: float | None,
) -> tuple[list[tuple[int, int]], float]:
    """Associate each transformed source with its nearest target.

    Target points may be matched repeatedly; exact distance ties resolve to
    the smallest target index because targets are scanned in ascending order
    with a strict comparison. Returns the retained pairs and the sum of
    their squared distances.
    """
    pairs: list[tuple[int, int]] = []
    squared_sum = 0.0
    for source_index, point in enumerate(transformed):
        best_index = 0
        best_distance = math.inf
        px, py, pz = point
        for target_index, candidate in enumerate(target):
            dx = px - candidate[0]
            dy = py - candidate[1]
            dz = pz - candidate[2]
            distance = dx * dx + dy * dy + dz * dz
            if distance < best_distance:
                best_distance = distance
                best_index = target_index
        if gate is None or best_distance <= gate * gate:
            pairs.append((source_index, best_index))
            squared_sum += best_distance
    return pairs, squared_sum


def _jacobi_eigen(
    matrix: Sequence[Sequence[float]],
) -> tuple[list[float], list[list[float]]]:
    """Eigen-decomposition of a real symmetric matrix via Jacobi rotations.

    Works for the small fixed sizes used here (3x3 and 4x4). Returns
    ``(eigenvalues, eigenvectors)`` where ``eigenvectors[k]`` is the unit
    eigenvector for ``eigenvalues[k]``; eigenvalues are descending.
    """
    size = len(matrix)
    a = [row[:] for row in matrix]
    v = [[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)]
    pairs = [(p, q) for p in range(size) for q in range(p + 1, size)]
    for _ in range(100):
        off = math.sqrt(sum(a[p][q] * a[p][q] for p, q in pairs))
        scale = max(abs(a[i][j]) for i in range(size) for j in range(size))
        if off <= 1e-15 * max(scale, 1.0):
            break
        for p, q in pairs:
            if a[p][q] == 0.0:
                continue
            # Numerically stable rotation that zeroes a[p][q].
            tau = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
            if tau >= 0.0:
                tangent = 1.0 / (tau + math.sqrt(1.0 + tau * tau))
            else:
                tangent = -1.0 / (-tau + math.sqrt(1.0 + tau * tau))
            cosine = 1.0 / math.sqrt(1.0 + tangent * tangent)
            sine = tangent * cosine
            for k in range(size):
                if k != p and k != q:
                    apk = a[p][k]
                    aqk = a[q][k]
                    a[p][k] = cosine * apk - sine * aqk
                    a[q][k] = sine * apk + cosine * aqk
                    a[k][p] = a[p][k]
                    a[k][q] = a[q][k]
            app = a[p][p]
            aqq = a[q][q]
            apq = a[p][q]
            a[p][p] = cosine * cosine * app - 2.0 * sine * cosine * apq + sine * sine * aqq
            a[q][q] = sine * sine * app + 2.0 * sine * cosine * apq + cosine * cosine * aqq
            a[p][q] = 0.0
            a[q][p] = 0.0
            for k in range(size):
                vkp = v[k][p]
                vkq = v[k][q]
                v[k][p] = cosine * vkp - sine * vkq
                v[k][q] = sine * vkp + cosine * vkq
    eigenvalues = [a[i][i] for i in range(size)]
    eigenvectors = [[v[k][index] for k in range(size)] for index in range(size)]
    order = sorted(range(size), key=lambda index: eigenvalues[index], reverse=True)
    ordered_values = [eigenvalues[index] for index in order]
    ordered_vectors = [eigenvectors[index] for index in order]
    return ordered_values, ordered_vectors


def _solve_rigid(
    source: Sequence[tuple[float, float, float]],
    target: Sequence[tuple[float, float, float]],
    pairs: Sequence[tuple[int, int]],
) -> Pose3:
    """Least-squares rigid transform for the given index pairs (Horn, 1987).

    The optimal unit quaternion is the dominant eigenvector of a symmetric
    4x4 matrix built from the paired cross-covariance; a unit quaternion can
    only represent a proper rotation, so the result never contains a mirror.
    Raises ValueError when fewer than three pairs survive or the geometry
    cannot uniquely determine a 3D rigid pose (collinear pairs, coincident
    centroids, or planar pairs related only by a reflection).
    """
    count = len(pairs)
    if count < 3:
        raise ValueError(
            f"at least 3 correspondences are required to constrain a rigid pose, got {count}"
        )

    source_centroid = [0.0, 0.0, 0.0]
    target_centroid = [0.0, 0.0, 0.0]
    for source_index, target_index in pairs:
        sp = source[source_index]
        tp = target[target_index]
        for axis in range(3):
            source_centroid[axis] += sp[axis]
            target_centroid[axis] += tp[axis]
    for axis in range(3):
        source_centroid[axis] /= count
        target_centroid[axis] /= count

    # s[i][j] = sum_p centered_source[p][i] * centered_target[p][j].
    s = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    for source_index, target_index in pairs:
        sp = source[source_index]
        tp = target[target_index]
        centered_source = tuple(sp[axis] - source_centroid[axis] for axis in range(3))
        centered_target = tuple(tp[axis] - target_centroid[axis] for axis in range(3))
        for i in range(3):
            for j in range(3):
                s[i][j] += centered_source[i] * centered_target[j]

    # Non-collinear centered geometry is enough to pin the rotation down:
    # the cross-covariance only needs rank 2 for a unique proper rotation
    # (three non-collinear points already span a plane with a definite pose).
    # Rank 1 means collinear pairs, whose rotation about the line is free.
    # Its squared singular values are the eigenvalues of S S^T.
    sst = [
        [sum(s[i][k] * s[j][k] for k in range(3)) for j in range(3)]
        for i in range(3)
    ]
    scatter_eigenvalues, _ = _jacobi_eigen(sst)
    largest_squared = scatter_eigenvalues[0]
    if largest_squared <= 0.0:
        raise ValueError(
            "correspondences are degenerate (all points coincide); "
            "a unique 3D rigid transform cannot be determined"
        )
    if scatter_eigenvalues[1] <= _SINGULAR_RANK_TOL * _SINGULAR_RANK_TOL * largest_squared:
        raise ValueError(
            "correspondences are degenerate (collinear); "
            "a unique 3D rigid transform cannot be determined"
        )

    sxx, sxy, sxz = s[0]
    syx, syy, syz = s[1]
    szx, szy, szz = s[2]
    trace = sxx + syy + szz
    # Horn's symmetric 4x4 matrix in (w, x, y, z) quaternion order. Its
    # dominant eigenvector is the rotation minimizing squared pair error.
    n = [
        [trace, syz - szy, szx - sxz, sxy - syx],
        [syz - szy, sxx - syy - szz, sxy + syx, szx + sxz],
        [szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy],
        [sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz],
    ]
    eigenvalues, eigenvectors = _jacobi_eigen(n)
    if not math.isfinite(eigenvalues[0]):
        raise ValueError("correspondences are degenerate; a rigid transform cannot be fit")
    # With planar (rank-2) pairs the rotation is unique unless the two planes
    # are mirror-related; that case leaves the top eigenvalue repeated (any
    # spin about the plane normal is optimal). Compare the gap relatively so
    # the check stays scale invariant.
    eigenvalue_gap = eigenvalues[0] - eigenvalues[1]
    if eigenvalue_gap <= _SINGULAR_RANK_TOL * abs(eigenvalues[0]):
        raise ValueError(
            "correspondences are degenerate (mirrored planar geometry); "
            "a unique 3D rigid transform cannot be determined"
        )
    quaternion = tuple(eigenvectors[0])

    # Translation from the centroids; build the rotation from the quaternion
    # directly instead of routing back through a matrix.
    w, qx, qy, qz = quaternion
    rotation = (
        (1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - w * qz), 2.0 * (qx * qz + w * qy)),
        (2.0 * (qx * qy + w * qz), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - w * qx)),
        (2.0 * (qx * qz - w * qy), 2.0 * (qy * qz + w * qx), 1.0 - 2.0 * (qx * qx + qy * qy)),
    )
    translation = tuple(
        target_centroid[axis]
        - sum(rotation[axis][j] * source_centroid[j] for j in range(3))
        for axis in range(3)
    )
    return Pose3(translation, quaternion)


def point_to_point_icp(
    source_points: Iterable[Iterable[float]],
    target_points: Iterable[Iterable[float]],
    initial_pose: Pose3 | None = None,
    max_iterations: int = 50,
    tolerance: float = 1e-6,
    max_correspondence_distance: float | None = None,
) -> ICPResult:
    """Align ``source_points`` onto ``target_points`` with point-to-point ICP.

    Args:
        source_points: At least three 3D points to align.
        target_points: At least three 3D points to align against.
        initial_pose: Initial source-to-target guess; defaults to identity.
        max_iterations: Maximum pose updates (positive integer, default 50).
        tolerance: Convergence threshold on the absolute RMSE change between
            iterations (finite positive float, default 1e-6).
        max_correspondence_distance: When given, pairs farther than this are
            discarded. None (default) keeps every pair.

    Returns:
        :class:`ICPResult` with the final pose, convergence flag, number of
        updates performed, final RMSE, and source-index-sorted pairs.

    Raises:
        TypeError: On non-iterable clouds, malformed points, non-real or
            boolean coordinates/parameters, or an ``initial_pose`` that is
            not a :class:`~mappilot.geometry.Pose3`.
        ValueError: On NaN/infinite coordinates, clouds with fewer than three
            points, non-positive iteration counts or tolerances, fewer than
            three gated pairs, or degenerate pair geometry.
    """
    source = _point_cloud(source_points, "source_points")
    target = _point_cloud(target_points, "target_points")
    if initial_pose is None:
        pose = Pose3.identity()
    elif not isinstance(initial_pose, Pose3):
        raise TypeError(
            f"initial_pose must be a Pose3, got {type(initial_pose).__name__}"
        )
    else:
        pose = initial_pose
    iteration_limit = _positive_int(max_iterations, "max_iterations")
    convergence_tolerance = _finite_positive_float(tolerance, "tolerance")
    gate = None
    if max_correspondence_distance is not None:
        gate = _finite_positive_float(
            max_correspondence_distance, "max_correspondence_distance"
        )

    # Initial association under the starting pose; its RMSE is the baseline
    # that the first pose update is compared against for convergence.
    transformed = pose.transform_points(source)
    pairs, squared_sum = _nearest_targets(transformed, target, gate)
    if len(pairs) < 3:
        raise ValueError(
            f"at least 3 correspondences are required to constrain a rigid pose, got {len(pairs)}"
        )
    rmse = math.sqrt(squared_sum / len(pairs))

    iterations = 0
    converged = False
    previous_rmse = rmse
    for _ in range(iteration_limit):
        increment = _solve_rigid(transformed, target, pairs)
        pose = increment.compose(pose)
        iterations += 1

        transformed = pose.transform_points(source)
        pairs, squared_sum = _nearest_targets(transformed, target, gate)
        if len(pairs) < 3:
            raise ValueError(
                "fewer than 3 correspondences remain after applying the "
                f"distance gate (got {len(pairs)})"
            )
        rmse = math.sqrt(squared_sum / len(pairs))

        if abs(previous_rmse - rmse) <= convergence_tolerance:
            converged = True
            break
        previous_rmse = rmse

    # The final RMSE and correspondences are recomputed from the final pose so
    # they are always mutually consistent, even when the budget ran out.
    transformed = pose.transform_points(source)
    pairs, squared_sum = _nearest_targets(transformed, target, gate)
    if len(pairs) < 3:
        raise ValueError(
            f"fewer than 3 final correspondences remain after the distance gate (got {len(pairs)})"
        )
    rmse = math.sqrt(squared_sum / len(pairs))
    correspondences = tuple(sorted(pairs, key=lambda pair: pair[0]))
    return ICPResult(pose, converged, iterations, rmse, correspondences)
