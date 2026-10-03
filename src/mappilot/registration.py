"""Point-to-point and point-to-plane ICP scan registration.

The :func:`point_to_point_icp` and :func:`point_to_plane_icp` entry points
estimate the rigid :class:`~mappilot.geometry.Pose3` that aligns a source
point cloud onto a target point cloud using the classic iterative closest
point loop:

1. Transform the source points with the current pose.
2. Match every transformed source point to its nearest target point (targets
   may be chosen repeatedly; exact distance ties resolve to the smallest target
   index). When a correspondence distance gate is given, pairs farther than
   the gate are dropped.
3. Solve the rigid transform minimizing the sum of squared pair distances
   (SVD of the centered cross-covariance), enforcing determinant +1 so the
   rotation can never contain a reflection. Point-to-plane instead solves
   the linearized least-squares system for the 6-DoF tangent increment that
   minimizes the squared signed projection of each pair's offset onto the
   target normal, and applies it through :meth:`Pose3.exp`.
4. Stop once the absolute RMSE change between two rounds is within tolerance.

This module is pure Python and depends only on the standard library plus
:mod:`mappilot.geometry`. Importing it never starts the HTTP service.

Validation policy mirrors :mod:`mappilot.geometry`: non-real scalars
(including booleans), non-iterable clouds, wrong point dimensions, and
parameters of the wrong type raise :class:`TypeError`; NaN or infinite
coordinates or parameter values, clouds with too few points (three for
point-to-point, six for point-to-plane), target-normal count mismatches,
zero-length normals, non-positive iteration limits/tolerances/gates, too
few valid correspondences, and correspondence configurations that cannot
uniquely constrain the 3D rigid pose (collinear point sets, parallel
normal fields, and the like) raise :class:`ValueError`.
"""

from __future__ import annotations

import math
import numbers
from typing import Iterable, NamedTuple

from .geometry import Pose3

__all__ = [
    "ICPResult",
    "PointToPlaneICPResult",
    "point_to_plane_icp",
    "point_to_point_icp",
]

# Machine epsilon used by the Jacobi SVD and the rank-deficiency test.
_EPS = 2.220446049250313e-16

# Upper bound on Jacobi sweeps; 3x3 matrices converge long before this.
_MAX_SWEEPS = 64

_AXES = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


class ICPResult(NamedTuple):
    """Outcome of :func:`point_to_point_icp`.

    Immutable. ``correspondences`` holds ``(source_index, target_index)``
    pairs ordered by ascending source index, recomputed from nearest
    neighbours under the returned ``pose``; ``rmse`` is the root mean squared
    Euclidean distance of exactly those pairs, so the two always agree.
    """

    pose: Pose3
    converged: bool
    iterations: int
    rmse: float
    correspondences: tuple[tuple[int, int], ...]


class PointToPlaneICPResult(NamedTuple):
    """Outcome of :func:`point_to_plane_icp`.

    Immutable. ``correspondences`` holds ``(source_index, target_index)``
    pairs ordered by ascending source index, recomputed from nearest
    neighbours under the returned ``pose``; ``rmse`` is the root mean
    squared signed projection error of exactly those pairs onto their
    target normals, so the two always agree.
    """

    pose: Pose3
    converged: bool
    iterations: int
    rmse: float
    correspondences: tuple[tuple[int, int], ...]


# -- validation -------------------------------------------------------------


def _real_scalar(value: object, name: str) -> float:
    """Coerce one real, finite, non-boolean scalar to float."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {result!r}")
    return result


def _point_cloud(
    value: object, name: str, minimum: int = 3
) -> tuple[tuple[float, float, float], ...]:
    """Validate a cloud of at least ``minimum`` 3D finite points."""
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be an iterable of 3D points")
    try:
        raw_points = list(value)  # type: ignore[call-overload]
    except TypeError:
        raise TypeError(f"{name} must be an iterable of 3D points") from None
    if len(raw_points) < minimum:
        raise ValueError(
            f"{name} must contain at least {minimum} points, got {len(raw_points)}"
        )

    points: list[tuple[float, float, float]] = []
    for index, raw_point in enumerate(raw_points):
        if isinstance(raw_point, (str, bytes, bytearray)):
            raise TypeError(f"{name}[{index}] must be a sequence of 3 real numbers")
        try:
            raw_coords = list(raw_point)  # type: ignore[call-overload]
        except TypeError:
            raise TypeError(
                f"{name}[{index}] must be a sequence of 3 real numbers"
            ) from None
        if len(raw_coords) != 3:
            raise TypeError(
                f"{name}[{index}] must have 3 coordinates, got {len(raw_coords)}"
            )
        point = tuple(
            _real_scalar(coord, f"{name}[{index}][{coord_index}]")
            for coord_index, coord in enumerate(raw_coords)
        )
        points.append(point)  # type: ignore[arg-type]
    return tuple(points)


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer, got {result}")
    return result


def _positive_float(value: object, name: str) -> float:
    result = _real_scalar(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be greater than zero, got {result}")
    return result


def _normal_field(
    value: object, name: str, expected_count: int
) -> tuple[tuple[float, float, float], ...]:
    """Validate one finite, non-zero 3D normal per target point, normalized.

    Each normal is divided by its own Euclidean length, so scaling any
    normal by a positive factor never changes the result.
    """
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be an iterable of 3D vectors")
    try:
        raw_normals = list(value)  # type: ignore[call-overload]
    except TypeError:
        raise TypeError(f"{name} must be an iterable of 3D vectors") from None
    if len(raw_normals) != expected_count:
        raise ValueError(
            f"{name} must contain one normal per target point, got "
            f"{len(raw_normals)} for {expected_count} target points"
        )

    normals: list[tuple[float, float, float]] = []
    for index, raw_normal in enumerate(raw_normals):
        if isinstance(raw_normal, (str, bytes, bytearray)):
            raise TypeError(f"{name}[{index}] must be a sequence of 3 real numbers")
        try:
            raw_coords = list(raw_normal)  # type: ignore[call-overload]
        except TypeError:
            raise TypeError(
                f"{name}[{index}] must be a sequence of 3 real numbers"
            ) from None
        if len(raw_coords) != 3:
            raise TypeError(
                f"{name}[{index}] must have 3 components, got {len(raw_coords)}"
            )
        normal = tuple(
            _real_scalar(coord, f"{name}[{index}][{coord_index}]")
            for coord_index, coord in enumerate(raw_coords)
        )
        length = math.sqrt(
            normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]
        )
        if length == 0.0:
            raise ValueError(f"{name}[{index}] must have non-zero length")
        normals.append(
            (normal[0] / length, normal[1] / length, normal[2] / length)
        )
    return tuple(normals)


# -- nearest neighbours -----------------------------------------------------


def _match(
    moved: tuple[tuple[float, float, float], ...],
    target: tuple[tuple[float, float, float], ...],
    gate_squared: float | None,
) -> list[tuple[int, int, float]]:
    """Nearest target per source, ascending source order.

    Returns ``(source_index, target_index, squared_distance)`` triples. Targets
    may be reused. Iterating targets in index order with a strict comparison
    makes exact ties resolve to the smallest target index. Pairs whose squared
    distance exceeds ``gate_squared`` are dropped (gate ``None`` keeps all).
    """
    pairs: list[tuple[int, int, float]] = []
    for source_index, source in enumerate(moved):
        sx, sy, sz = source
        best_index = -1
        best_distance = math.inf
        for target_index, point in enumerate(target):
            dx = sx - point[0]
            dy = sy - point[1]
            dz = sz - point[2]
            distance = dx * dx + dy * dy + dz * dz
            if distance < best_distance:
                best_distance = distance
                best_index = target_index
        if gate_squared is None or best_distance <= gate_squared:
            pairs.append((source_index, best_index, best_distance))
    return pairs


def _rmse_of(pairs: list[tuple[int, int, float]]) -> float:
    return math.sqrt(sum(distance for _, _, distance in pairs) / len(pairs))


# -- 3x3 SVD (one-sided Jacobi, standard library only) ----------------------


def _det3(matrix: tuple[tuple[float, float, float], ...]) -> float:
    (a00, a01, a02), (a10, a11, a12), (a20, a21, a22) = matrix
    return (
        a00 * (a11 * a22 - a12 * a21)
        - a01 * (a10 * a22 - a12 * a20)
        + a02 * (a10 * a21 - a11 * a20)
    )


def _svd3(
    a: tuple[tuple[float, float, float], ...]
) -> tuple[tuple[tuple[float, float, float], ...], tuple[float, float, float],
           tuple[tuple[float, float, float], ...]]:
    """SVD of a 3x3 matrix: ``A = U diag(s) V^T`` with singular values descending.

    One-sided Jacobi rotations diagonalize ``A^T A``; singular vectors for zero
    singular values are completed against the fixed coordinate axes so the
    result stays deterministic.
    """
    columns = [[a[row][col] for row in range(3)] for col in range(3)]
    v = [[1.0 if row == col else 0.0 for col in range(3)] for row in range(3)]

    for _ in range(_MAX_SWEEPS):
        converged = True
        for p, q in ((0, 1), (0, 2), (1, 2)):
            col_p = columns[p]
            col_q = columns[q]
            alpha = sum(x * x for x in col_p)
            beta = sum(x * x for x in col_q)
            gamma = sum(col_p[k] * col_q[k] for k in range(3))
            if gamma == 0.0 or abs(gamma) <= _EPS * math.sqrt(alpha * beta):
                continue
            converged = False
            zeta = (beta - alpha) / (2.0 * gamma)
            if zeta >= 0.0:
                tau = 1.0 / (zeta + math.sqrt(1.0 + zeta * zeta))
            else:
                tau = -1.0 / (-zeta + math.sqrt(1.0 + zeta * zeta))
            cosine = 1.0 / math.sqrt(1.0 + tau * tau)
            sine = tau * cosine
            for k in range(3):
                x = col_p[k]
                y = col_q[k]
                col_p[k] = cosine * x - sine * y
                col_q[k] = sine * x + cosine * y
            for row in range(3):
                x = v[row][p]
                y = v[row][q]
                v[row][p] = cosine * x - sine * y
                v[row][q] = sine * x + cosine * y
        if converged:
            break

    raw_singular = tuple(math.sqrt(sum(x * x for x in column)) for column in columns)
    largest = raw_singular[0]
    for value in raw_singular[1:]:
        if value > largest:
            largest = value
    # Jacobi leaves O(eps) noise in null-space columns; treat anything near
    # that floor as a zero singular value and complete its vectors instead.
    zero_tolerance = largest * 32.0 * _EPS if largest > 0.0 else 0.0
    order = sorted(range(3), key=raw_singular.__getitem__, reverse=True)

    def complete_nullspace(known: list[list[float]]) -> list[list[float]]:
        added: list[list[float]] = []
        for axis in _AXES:
            if len(known) + len(added) == 3:
                break
            vector = list(axis)
            for reference in known + added:
                projection = sum(vector[k] * reference[k] for k in range(3))
                for k in range(3):
                    vector[k] -= projection * reference[k]
            norm = math.sqrt(sum(x * x for x in vector))
            if norm > 0.5:
                added.append([x / norm for x in vector])
        return added

    known_u: dict[int, list[float]] = {}
    known_v: dict[int, list[float]] = {}
    u_vectors: list[list[float]] = []
    v_vectors: list[list[float]] = []
    for slot, original in enumerate(order):
        value = raw_singular[original]
        if value > zero_tolerance:
            known_u[slot] = [columns[original][row] / value for row in range(3)]
            v_column = [v[row][original] for row in range(3)]
            v_norm = math.sqrt(sum(x * x for x in v_column))
            known_v[slot] = [x / v_norm for x in v_column]
            u_vectors.append(known_u[slot])
            v_vectors.append(known_v[slot])

    extra_u = iter(complete_nullspace(u_vectors))
    extra_v = iter(complete_nullspace(v_vectors))
    u_basis = [
        known_u[slot] if slot in known_u else next(extra_u) for slot in range(3)
    ]
    v_basis = [
        known_v[slot] if slot in known_v else next(extra_v) for slot in range(3)
    ]

    u = tuple(tuple(u_basis[col][row] for col in range(3)) for row in range(3))
    vt = tuple(
        tuple(v_basis[col][row] for row in range(3)) for col in range(3)
    )
    ordered_singular = tuple(raw_singular[index] for index in order)
    return u, ordered_singular, vt


# -- rigid transform solve --------------------------------------------------


def _solve_rigid(
    moved: tuple[tuple[float, float, float], ...],
    target: tuple[tuple[float, float, float], ...],
    pairs: list[tuple[int, int, float]],
) -> Pose3:
    """Minimize squared pair distances; return the rigid transform ``moved -> target``."""
    count = len(pairs)
    mean_s = [0.0, 0.0, 0.0]
    mean_t = [0.0, 0.0, 0.0]
    for source_index, target_index, _ in pairs:
        source = moved[source_index]
        point = target[target_index]
        for k in range(3):
            mean_s[k] += source[k]
            mean_t[k] += point[k]
    mean_s = [value / count for value in mean_s]
    mean_t = [value / count for value in mean_t]

    cross = [[0.0, 0.0, 0.0] for _ in range(3)]
    for source_index, target_index, _ in pairs:
        source = moved[source_index]
        point = target[target_index]
        centered_s = [source[k] - mean_s[k] for k in range(3)]
        centered_t = [point[k] - mean_t[k] for k in range(3)]
        for i in range(3):
            row = cross[i]
            cs_i = centered_s[i]
            for j in range(3):
                row[j] += cs_i * centered_t[j]

    u, singular, vt = _svd3(tuple(tuple(row) for row in cross))

    # Rank 1 (collinear or worse) leaves a rotation about the line free; the
    # 3D rigid pose is not unique. Rank 2 is still fully constrained.
    rank_tolerance = singular[0] * max(3, count) * _EPS
    if singular[1] <= rank_tolerance:
        raise ValueError(
            "correspondence points are too degenerate (collinear) to uniquely "
            "determine a 3D rigid pose"
        )

    # H = U diag(s) V^T minimizes with R = V diag(1, 1, det(V U^T)) U^T; the
    # sign flip rules out a reflection.
    v_matrix = tuple(tuple(vt[col][row] for col in range(3)) for row in range(3))
    determinant = _det3(v_matrix) * _det3(u)
    reflection_correction = (1.0, 1.0, determinant)
    rotation = tuple(
        tuple(
            sum(v_matrix[row][k] * reflection_correction[k] * u[col][k]
                for k in range(3))
            for col in range(3)
        )
        for row in range(3)
    )
    translation = tuple(
        mean_t[k] - sum(rotation[k][j] * mean_s[j] for j in range(3))
        for k in range(3)
    )
    return Pose3.from_matrix(
        (
            (rotation[0][0], rotation[0][1], rotation[0][2], translation[0]),
            (rotation[1][0], rotation[1][1], rotation[1][2], translation[1]),
            (rotation[2][0], rotation[2][1], rotation[2][2], translation[2]),
            (0.0, 0.0, 0.0, 1.0),
        )
    )


# -- point-to-plane solve ---------------------------------------------------


def _projection_residual(
    moved: tuple[float, float, float],
    point: tuple[float, float, float],
    normal: tuple[float, float, float],
) -> float:
    """Signed projection of ``moved - point`` onto the target ``normal``."""
    return (
        normal[0] * (moved[0] - point[0])
        + normal[1] * (moved[1] - point[1])
        + normal[2] * (moved[2] - point[2])
    )


def _projection_rmse(
    moved: tuple[tuple[float, float, float], ...],
    target: tuple[tuple[float, float, float], ...],
    normals: tuple[tuple[float, float, float], ...],
    pairs: list[tuple[int, int, float]],
) -> float:
    total = 0.0
    for source_index, target_index, _ in pairs:
        residual = _projection_residual(
            moved[source_index], target[target_index], normals[target_index]
        )
        total += residual * residual
    return math.sqrt(total / len(pairs))


def _solve_linear6(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """Solve a 6x6 system by Gaussian elimination with partial pivoting.

    Raises :class:`ValueError` when the normal constraints cannot uniquely
    determine the 6-DoF pose increment (the system is singular to within
    floating-point noise). Pivoting is deterministic, so identical inputs
    always produce identical increments.
    """
    size = 6
    augmented = [row[:] + [rhs[index]] for index, row in enumerate(matrix)]
    scale = 0.0
    for row in augmented:
        for value in row[:size]:
            if abs(value) > scale:
                scale = abs(value)
    if scale == 0.0:
        raise ValueError(
            "target-normal constraints cannot uniquely determine the "
            "6-DoF pose increment"
        )
    # A singular system leaves pivots at the O(eps * scale) noise floor.
    singular_tolerance = scale * 64.0 * _EPS

    for col in range(size):
        pivot = col
        for candidate in range(col + 1, size):
            if abs(augmented[candidate][col]) > abs(augmented[pivot][col]):
                pivot = candidate
        if abs(augmented[pivot][col]) <= singular_tolerance:
            raise ValueError(
                "target-normal constraints cannot uniquely determine the "
                "6-DoF pose increment"
            )
        if pivot != col:
            augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        pivot_value = augmented[col][col]
        for row in range(col + 1, size):
            factor = augmented[row][col] / pivot_value
            if factor == 0.0:
                continue
            augmented[row][col] = 0.0
            for k in range(col + 1, size + 1):
                augmented[row][k] -= factor * augmented[col][k]

    result = [0.0] * size
    for row in range(size - 1, -1, -1):
        total = augmented[row][size]
        for k in range(row + 1, size):
            total -= augmented[row][k] * result[k]
        result[row] = total / augmented[row][row]
    return result


def _solve_point_to_plane(
    moved: tuple[tuple[float, float, float], ...],
    target: tuple[tuple[float, float, float], ...],
    normals: tuple[tuple[float, float, float], ...],
    pairs: list[tuple[int, int, float]],
) -> Pose3:
    """Tangent increment minimizing squared signed projection errors.

    Linearizes the residual ``n . (R p + t - q)`` of each pair around the
    current pose with the left-perturbation model ``p -> p + w x p + v``
    and solves the 6x6 normal equations for ``(w, v)``; the returned pose
    is :meth:`Pose3.exp` of that increment, to be composed on the left.
    """
    normal_matrix = [[0.0] * 6 for _ in range(6)]
    normal_rhs = [0.0] * 6
    for source_index, target_index, _ in pairs:
        source = moved[source_index]
        point = target[target_index]
        normal = normals[target_index]
        # Row of the Jacobian: (p x n, n) against (w, v).
        cross = (
            source[1] * normal[2] - source[2] * normal[1],
            source[2] * normal[0] - source[0] * normal[2],
            source[0] * normal[1] - source[1] * normal[0],
        )
        row = (cross[0], cross[1], cross[2], normal[0], normal[1], normal[2])
        residual = _projection_residual(source, point, normal)
        for i in range(6):
            row_i = row[i]
            normal_rhs[i] -= row_i * residual
            matrix_row = normal_matrix[i]
            for j in range(6):
                matrix_row[j] += row_i * row[j]
    increment = _solve_linear6(normal_matrix, normal_rhs)
    return Pose3.exp(tuple(increment))


# -- public entry points ----------------------------------------------------


def point_to_point_icp(
    source_points: Iterable[Iterable[float]],
    target_points: Iterable[Iterable[float]],
    initial_pose: Pose3 | None = None,
    max_iterations: int = 50,
    tolerance: float = 1e-6,
    max_correspondence_distance: float | None = None,
) -> ICPResult:
    """Align ``source_points`` onto ``target_points`` with point-to-point ICP.

    Parameters
    ----------
    source_points, target_points:
        Iterables of at least three ``(x, y, z)`` points each. Coordinates must
        be finite real numbers (booleans rejected).
    initial_pose:
        Pose applied to the source before the first round; defaults to identity.
    max_iterations:
        Maximum number of pose updates; a positive integer, default 50.
    tolerance:
        Convergence threshold on the absolute RMSE change between consecutive
        rounds; a finite positive number, default 1e-6.
    max_correspondence_distance:
        When given (finite positive), pairs farther apart than this are
        discarded; ``None`` (default) keeps every nearest pair regardless of
        distance.

    Returns
    -------
    ICPResult
        Never raises on non-convergence: exhausting ``max_iterations`` returns
        the last pose with ``converged=False``. The reported correspondences
        and RMSE are recomputed from nearest neighbours under the returned
        pose, with correspondences ordered by ascending source index.
    """
    source = _point_cloud(source_points, "source_points")
    target = _point_cloud(target_points, "target_points")
    if initial_pose is None:
        pose = Pose3.identity()
    elif isinstance(initial_pose, Pose3):
        pose = initial_pose
    else:
        raise TypeError(
            f"initial_pose must be a Pose3, got {type(initial_pose).__name__}"
        )
    iteration_limit = _positive_int(max_iterations, "max_iterations")
    convergence_tolerance = _positive_float(tolerance, "tolerance")
    gate = max_correspondence_distance
    if gate is not None:
        gate_squared: float | None = _positive_float(
            gate, "max_correspondence_distance"
        ) ** 2
    else:
        gate_squared = None

    updates = 0
    previous_rmse: float | None = None
    converged = False

    for _ in range(iteration_limit):
        moved = pose.transform_points(source)
        pairs = _match(moved, target, gate_squared)
        if len(pairs) < 3:
            raise ValueError(
                "point-to-point ICP requires at least three valid "
                f"correspondences, got {len(pairs)}"
            )
        round_rmse = _rmse_of(pairs)
        if (
            previous_rmse is not None
            and abs(round_rmse - previous_rmse) <= convergence_tolerance
        ):
            converged = True
            break
        delta = _solve_rigid(moved, target, pairs)
        pose = delta.compose(pose)
        updates += 1
        previous_rmse = round_rmse

    # Recompute the reported pairs and RMSE from the final pose so they can
    # never describe a stale intermediate alignment.
    final_moved = pose.transform_points(source)
    final_pairs = _match(final_moved, target, gate_squared)
    if len(final_pairs) < 3:
        raise ValueError(
            "point-to-point ICP requires at least three valid "
            f"correspondences, got {len(final_pairs)}"
        )
    correspondences = tuple(
        (source_index, target_index) for source_index, target_index, _ in final_pairs
    )
    final_rmse = _rmse_of(final_pairs)
    return ICPResult(pose, converged, updates, final_rmse, correspondences)


def point_to_plane_icp(
    source_points: Iterable[Iterable[float]],
    target_points: Iterable[Iterable[float]],
    target_normals: Iterable[Iterable[float]],
    initial_pose: Pose3 | None = None,
    max_iterations: int = 50,
    tolerance: float = 1e-6,
    max_correspondence_distance: float | None = None,
) -> PointToPlaneICPResult:
    """Align ``source_points`` onto ``target_points`` with point-to-plane ICP.

    Parameters
    ----------
    source_points, target_points:
        Iterables of at least six ``(x, y, z)`` points each. Coordinates
        must be finite real numbers (booleans rejected).
    target_normals:
        One finite, non-zero ``(nx, ny, nz)`` normal per target point.
        Each normal is normalized by its own length before use, so scaling
        a normal by a positive factor never changes the result.
    initial_pose:
        Pose applied to the source before the first round; defaults to identity.
    max_iterations:
        Maximum number of pose updates; a positive integer, default 50.
    tolerance:
        Convergence threshold on the absolute RMSE change between consecutive
        rounds; a finite positive number, default 1e-6.
    max_correspondence_distance:
        When given (finite positive), pairs farther apart than this are
        discarded; ``None`` (default) keeps every nearest pair regardless of
        distance.

    Returns
    -------
    PointToPlaneICPResult
        Never raises on non-convergence: exhausting ``max_iterations``
        returns the last pose with ``converged=False``. The reported
        correspondences and RMSE are recomputed from nearest neighbours
        under the returned pose, with correspondences ordered by ascending
        source index; ``rmse`` is the root mean squared signed projection
        error of those pairs onto their target normals.

    Raises
    ------
    ValueError
        If fewer than six valid correspondences remain, or the target
        normals cannot uniquely constrain the 6-DoF pose increment (for
        example a planar patch with a parallel normal field).
    """
    source = _point_cloud(source_points, "source_points", minimum=6)
    target = _point_cloud(target_points, "target_points", minimum=6)
    normals = _normal_field(target_normals, "target_normals", len(target))
    if initial_pose is None:
        pose = Pose3.identity()
    elif isinstance(initial_pose, Pose3):
        pose = initial_pose
    else:
        raise TypeError(
            f"initial_pose must be a Pose3, got {type(initial_pose).__name__}"
        )
    iteration_limit = _positive_int(max_iterations, "max_iterations")
    convergence_tolerance = _positive_float(tolerance, "tolerance")
    gate = max_correspondence_distance
    if gate is not None:
        gate_squared: float | None = _positive_float(
            gate, "max_correspondence_distance"
        ) ** 2
    else:
        gate_squared = None

    updates = 0
    previous_rmse: float | None = None
    converged = False

    for _ in range(iteration_limit):
        moved = pose.transform_points(source)
        pairs = _match(moved, target, gate_squared)
        if len(pairs) < 6:
            raise ValueError(
                "point-to-plane ICP requires at least six valid "
                f"correspondences, got {len(pairs)}"
            )
        round_rmse = _projection_rmse(moved, target, normals, pairs)
        if (
            previous_rmse is not None
            and abs(round_rmse - previous_rmse) <= convergence_tolerance
        ):
            converged = True
            break
        delta = _solve_point_to_plane(moved, target, normals, pairs)
        pose = delta.compose(pose)
        updates += 1
        previous_rmse = round_rmse

    # Recompute the reported pairs and RMSE from the final pose so they can
    # never describe a stale intermediate alignment.
    final_moved = pose.transform_points(source)
    final_pairs = _match(final_moved, target, gate_squared)
    if len(final_pairs) < 6:
        raise ValueError(
            "point-to-plane ICP requires at least six valid "
            f"correspondences, got {len(final_pairs)}"
        )
    correspondences = tuple(
        (source_index, target_index) for source_index, target_index, _ in final_pairs
    )
    final_rmse = _projection_rmse(final_moved, target, normals, final_pairs)
    return PointToPlaneICPResult(pose, converged, updates, final_rmse, correspondences)
