"""Pose-graph optimization over :class:`~mappilot.geometry.Pose3` nodes.

The :func:`optimize_pose_graph` entry point takes initial pose guesses plus
relative-pose constraints and refines the poses with Gauss-Newton iterations
on SE(3) so the whole trajectory becomes globally consistent. Each constraint
``(from_id, to_id, measurement, information)`` contributes the residual

    e = log(measurement^-1 * pose_from^-1 * pose_to)

(a 6D tangent vector ordered like :meth:`Pose3.log`: rotation first, then
translation) and the total error is the sum of the quadratic forms
``e^T information e`` over all constraints. Free poses are updated by left
perturbation, ``pose <- Pose3.exp(delta) * pose``; poses listed in
``fixed_node_ids`` are never touched, so every connected component of the
constraint graph must contain at least one fixed node to anchor the gauge
freedom.

This module is pure Python and depends only on the standard library plus
:mod:`mappilot.geometry`. Importing it never starts the HTTP service.

Validation policy mirrors :mod:`mappilot.geometry` and
:mod:`mappilot.registration`: non-iterable inputs, non-integer (or boolean)
node IDs, non-:class:`Pose3` poses, malformed constraint tuples, wrongly
shaped information matrices, non-real (or boolean) matrix elements, and
wrongly typed ``max_iterations``/``tolerance`` raise :class:`TypeError`;
empty or duplicate node IDs, constraints referencing unknown nodes, unknown
or empty fixed sets, connected components without a fixed node, non-finite
matrix elements, information matrices that are asymmetric (absolute
tolerance 1e-12) or not positive definite, and non-positive or non-finite
``max_iterations``/``tolerance`` raise :class:`ValueError`.
"""

from __future__ import annotations

import math
import numbers
from typing import Iterable, NamedTuple

from .geometry import Pose3

__all__ = ["PoseGraphResult", "optimize_pose_graph"]

# Absolute tolerance for the symmetry check on information matrices.
_SYMMETRY_TOLERANCE = 1e-12

# Finite-difference step for the residual Jacobians. Central differences at
# this step keep the Jacobian error near 1e-10, far below the convergence
# tolerances the optimizer is meant to reach.
_JACOBIAN_STEP = 1e-6

# Relative diagonal jitter used to retry a numerically singular normal-equations
# factorization (e.g. a free node constrained only by self-loops).
_REGULARIZATION_START = 1e-9
_REGULARIZATION_ATTEMPTS = 6


class PoseGraphResult(NamedTuple):
    """Outcome of :func:`optimize_pose_graph`.

    Immutable. ``poses`` holds ``(id, Pose3)`` pairs in the input node order,
    with fixed poses value-identical to their inputs. ``initial_error`` and
    ``final_error`` are the total quadratic-form errors recomputed from the
    input poses and the returned poses respectively.
    """

    poses: tuple[tuple[int, Pose3], ...]
    converged: bool
    iterations: int
    initial_error: float
    final_error: float


# -- validation -------------------------------------------------------------


def _node_id(value: object, name: str) -> int:
    """Coerce one non-boolean integer node ID."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    return int(value)


def _nodes(value: object) -> tuple[tuple[int, Pose3], ...]:
    """Validate the ordered ``(id, Pose3)`` node sequence."""
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError("nodes must be an iterable of (id, Pose3) pairs")
    try:
        raw_nodes = list(value)  # type: ignore[call-overload]
    except TypeError:
        raise TypeError("nodes must be an iterable of (id, Pose3) pairs") from None
    if not raw_nodes:
        raise ValueError("nodes must contain at least one node")

    nodes: list[tuple[int, Pose3]] = []
    seen: set[int] = set()
    for index, raw_node in enumerate(raw_nodes):
        if isinstance(raw_node, (str, bytes, bytearray)):
            raise TypeError(f"nodes[{index}] must be a (id, Pose3) pair")
        try:
            pair = tuple(raw_node)  # type: ignore[call-overload]
        except TypeError:
            raise TypeError(f"nodes[{index}] must be a (id, Pose3) pair") from None
        if len(pair) != 2:
            raise TypeError(
                f"nodes[{index}] must have 2 elements (id, pose), got {len(pair)}"
            )
        node_id = _node_id(pair[0], f"nodes[{index}][0]")
        pose = pair[1]
        if not isinstance(pose, Pose3):
            raise TypeError(
                f"nodes[{index}][1] must be a Pose3, got {type(pose).__name__}"
            )
        if node_id in seen:
            raise ValueError(f"duplicate node id {node_id}")
        seen.add(node_id)
        nodes.append((node_id, pose))
    return tuple(nodes)


def _fixed_node_ids(value: object, known: set[int]) -> frozenset[int]:
    """Validate the non-empty set of fixed node IDs."""
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError("fixed_node_ids must be an iterable of integer node ids")
    try:
        raw_ids = list(value)  # type: ignore[call-overload]
    except TypeError:
        raise TypeError(
            "fixed_node_ids must be an iterable of integer node ids"
        ) from None
    fixed = frozenset(
        _node_id(raw_id, f"fixed_node_ids[{index}]")
        for index, raw_id in enumerate(raw_ids)
    )
    if not fixed:
        raise ValueError("fixed_node_ids must not be empty")
    for node_id in fixed:
        if node_id not in known:
            raise ValueError(f"fixed node id {node_id} does not match any node")
    return fixed


def _information_matrix(value: object, name: str) -> tuple[tuple[float, ...], ...]:
    """Validate a symmetric positive definite 6x6 information matrix."""
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be a 6x6 nested sequence of real numbers")
    try:
        rows = tuple(value)  # type: ignore[call-overload]
    except TypeError:
        raise TypeError(
            f"{name} must be a 6x6 nested sequence of real numbers"
        ) from None
    if len(rows) != 6:
        raise TypeError(f"{name} must have 6 rows, got {len(rows)}")

    grid: list[tuple[float, ...]] = []
    for row_index, row in enumerate(rows):
        if isinstance(row, (str, bytes, bytearray)):
            raise TypeError(f"{name}[{row_index}] must be a sequence of 6 real numbers")
        try:
            items = tuple(row)  # type: ignore[call-overload]
        except TypeError:
            raise TypeError(
                f"{name}[{row_index}] must be a sequence of 6 real numbers"
            ) from None
        if len(items) != 6:
            raise TypeError(
                f"{name}[{row_index}] must have 6 elements, got {len(items)}"
            )
        entries: list[float] = []
        for column_index, item in enumerate(items):
            if isinstance(item, bool) or not isinstance(item, numbers.Real):
                raise TypeError(
                    f"{name}[{row_index}][{column_index}] must be a real number, "
                    f"got {type(item).__name__}"
                )
            element = float(item)
            if not math.isfinite(element):
                raise ValueError(
                    f"{name}[{row_index}][{column_index}] must be finite, "
                    f"got {element!r}"
                )
            entries.append(element)
        grid.append(tuple(entries))

    for i in range(6):
        for j in range(i + 1, 6):
            if abs(grid[i][j] - grid[j][i]) > _SYMMETRY_TOLERANCE:
                raise ValueError(f"{name} must be symmetric within 1e-12")
    if not _is_positive_definite(grid):
        raise ValueError(f"{name} must be positive definite")
    return tuple(grid)


def _is_positive_definite(matrix: list[tuple[float, ...]] | tuple[tuple[float, ...], ...]) -> bool:
    """Cholesky test: the matrix is positive definite iff it factors."""
    size = len(matrix)
    lower = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(i + 1):
            total = matrix[i][j]
            for k in range(j):
                total -= lower[i][k] * lower[j][k]
            if i == j:
                if total <= 0.0:
                    return False
                lower[i][j] = math.sqrt(total)
            else:
                lower[i][j] = total / lower[j][j]
    return True


class _Constraint(NamedTuple):
    """One validated relative-pose constraint."""

    from_id: int
    to_id: int
    measurement: Pose3
    information: tuple[tuple[float, ...], ...]


def _constraints(value: object, known: set[int]) -> tuple[_Constraint, ...]:
    """Validate the constraint sequence."""
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(
            "constraints must be an iterable of "
            "(from_id, to_id, measurement, information) tuples"
        )
    try:
        raw_constraints = list(value)  # type: ignore[call-overload]
    except TypeError:
        raise TypeError(
            "constraints must be an iterable of "
            "(from_id, to_id, measurement, information) tuples"
        ) from None

    constraints: list[_Constraint] = []
    for index, raw_constraint in enumerate(raw_constraints):
        name = f"constraints[{index}]"
        if isinstance(raw_constraint, (str, bytes, bytearray)):
            raise TypeError(
                f"{name} must be a (from_id, to_id, measurement, information) tuple"
            )
        try:
            parts = tuple(raw_constraint)  # type: ignore[call-overload]
        except TypeError:
            raise TypeError(
                f"{name} must be a (from_id, to_id, measurement, information) tuple"
            ) from None
        if len(parts) != 4:
            raise TypeError(
                f"{name} must have 4 elements (from_id, to_id, measurement, "
                f"information), got {len(parts)}"
            )
        from_id = _node_id(parts[0], f"{name}[0]")
        to_id = _node_id(parts[1], f"{name}[1]")
        measurement = parts[2]
        if not isinstance(measurement, Pose3):
            raise TypeError(
                f"{name}[2] must be a Pose3, got {type(measurement).__name__}"
            )
        information = _information_matrix(parts[3], f"{name}[3]")
        for endpoint in (from_id, to_id):
            if endpoint not in known:
                raise ValueError(
                    f"{name} references unknown node id {endpoint}"
                )
        constraints.append(_Constraint(from_id, to_id, measurement, information))
    return tuple(constraints)


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer, got {result}")
    return result


def _positive_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {result!r}")
    if result <= 0.0:
        raise ValueError(f"{name} must be greater than zero, got {result}")
    return result


def _check_components_fixed(
    nodes: tuple[tuple[int, Pose3], ...],
    constraints: tuple[_Constraint, ...],
    fixed: frozenset[int],
) -> None:
    """Every connected component of the constraint graph needs a fixed node."""
    adjacency: dict[int, list[int]] = {node_id: [] for node_id, _ in nodes}
    for constraint in constraints:
        adjacency[constraint.from_id].append(constraint.to_id)
        adjacency[constraint.to_id].append(constraint.from_id)

    visited: set[int] = set()
    for node_id, _ in nodes:
        if node_id in visited:
            continue
        stack = [node_id]
        component_has_fixed = False
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            if current in fixed:
                component_has_fixed = True
            stack.extend(adjacency[current])
        if not component_has_fixed:
            raise ValueError(
                "every connected component of the pose graph must contain at "
                "least one fixed node"
            )


# -- error evaluation ---------------------------------------------------------


def _residual(
    constraint: _Constraint, poses: dict[int, Pose3]
) -> tuple[float, float, float, float, float, float]:
    """``log(measurement^-1 * pose_from^-1 * pose_to)`` as a 6-vector."""
    return (
        constraint.measurement.inverse()
        .compose(poses[constraint.from_id].inverse().compose(poses[constraint.to_id]))
        .log()
    )


def _total_error(constraints: tuple[_Constraint, ...], poses: dict[int, Pose3]) -> float:
    """Sum of ``e^T information e`` over all constraints."""
    total = 0.0
    for constraint in constraints:
        error = _residual(constraint, poses)
        information = constraint.information
        quadratic = 0.0
        for i in range(6):
            row = information[i]
            error_i = error[i]
            for j in range(6):
                quadratic += error_i * row[j] * error[j]
        total += quadratic
    return total


# -- linear algebra (dense, standard library only) ----------------------------


def _cholesky_factor(matrix: list[list[float]]) -> list[list[float]] | None:
    """Lower Cholesky factor of a symmetric matrix, or None if not definite."""
    size = len(matrix)
    lower = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(i + 1):
            total = matrix[i][j]
            for k in range(j):
                total -= lower[i][k] * lower[j][k]
            if i == j:
                if total <= 0.0:
                    return None
                lower[i][j] = math.sqrt(total)
            else:
                lower[i][j] = total / lower[j][j]
    return lower


def _cholesky_solve(lower: list[list[float]], rhs: list[float]) -> list[float]:
    """Solve ``L L^T x = rhs`` given the lower factor ``L``."""
    size = len(lower)
    y = [0.0] * size
    for i in range(size):
        total = rhs[i]
        for k in range(i):
            total -= lower[i][k] * y[k]
        y[i] = total / lower[i][i]
    x = [0.0] * size
    for i in range(size - 1, -1, -1):
        total = y[i]
        for k in range(i + 1, size):
            total -= lower[k][i] * x[k]
        x[i] = total / lower[i][i]
    return x


# -- Gauss-Newton step ---------------------------------------------------------


def _gauss_newton_step(
    constraints: tuple[_Constraint, ...],
    poses: dict[int, Pose3],
    free_ids: tuple[int, ...],
) -> list[float]:
    """One Gauss-Newton increment for the free poses (left-perturbation).

    Jacobians of the residuals with respect to left perturbations
    ``pose <- Pose3.exp(delta) * pose`` are computed by central finite
    differences; the normal equations are solved by Cholesky factorization
    with a small diagonal regularization retry for numerically singular
    configurations.
    """
    offsets = {node_id: 6 * slot for slot, node_id in enumerate(free_ids)}
    size = 6 * len(free_ids)
    hessian = [[0.0] * size for _ in range(size)]
    gradient = [0.0] * size

    for constraint in constraints:
        baseline = _residual(constraint, poses)
        information = constraint.information
        # Weighted residual, so the normal equations accumulate
        # J^T Omega J and J^T Omega e directly.
        weighted_baseline = [
            sum(information[i][j] * baseline[j] for j in range(6)) for i in range(6)
        ]
        # Jacobian column blocks for each free endpoint of this constraint.
        blocks: list[tuple[int, list[list[float]]]] = []
        for node_id in (constraint.from_id, constraint.to_id):
            if node_id not in offsets:
                continue
            offset = offsets[node_id]
            original = poses[node_id]
            columns: list[list[float]] = []
            for axis in range(6):
                step = [0.0] * 6
                step[axis] = _JACOBIAN_STEP
                poses[node_id] = Pose3.exp(step).compose(original)
                forward = _residual(constraint, poses)
                step[axis] = -_JACOBIAN_STEP
                poses[node_id] = Pose3.exp(step).compose(original)
                backward = _residual(constraint, poses)
                poses[node_id] = original
                columns.append(
                    [
                        (forward[k] - backward[k]) / (2.0 * _JACOBIAN_STEP)
                        for k in range(6)
                    ]
                )
            blocks.append((offset, columns))
        weighted_blocks = [
            (
                offset,
                [
                    [sum(information[i][j] * column[j] for j in range(6)) for i in range(6)]
                    for column in columns
                ],
            )
            for offset, columns in blocks
        ]
        for (offset_a, columns_a), (_, weighted_a) in zip(blocks, weighted_blocks):
            for a in range(6):
                column_a = columns_a[a]
                gradient[offset_a + a] += sum(
                    column_a[k] * weighted_baseline[k] for k in range(6)
                )
                row = hessian[offset_a + a]
                for offset_b, weighted_b in weighted_blocks:
                    for b in range(6):
                        row[offset_b + b] += sum(
                            column_a[k] * weighted_b[b][k] for k in range(6)
                        )

    rhs = [-value for value in gradient]
    jitter = 0.0
    largest_diagonal = max((hessian[i][i] for i in range(size)), default=0.0)
    scale = max(largest_diagonal, 1.0)
    for _ in range(_REGULARIZATION_ATTEMPTS):
        damped = [row[:] for row in hessian]
        if jitter > 0.0:
            for i in range(size):
                damped[i][i] += jitter
        lower = _cholesky_factor(damped)
        if lower is not None:
            return _cholesky_solve(lower, rhs)
        jitter = _REGULARIZATION_START * scale if jitter == 0.0 else jitter * 10.0
    raise ValueError(
        "normal equations are singular; the pose graph is under-constrained"
    )


# -- public entry point -------------------------------------------------------


def optimize_pose_graph(
    nodes: Iterable[tuple[int, Pose3]],
    constraints: Iterable[tuple[int, int, Pose3, Iterable[Iterable[float]]]],
    fixed_node_ids: Iterable[int],
    max_iterations: int = 50,
    tolerance: float = 1e-6,
) -> PoseGraphResult:
    """Optimize pose-graph nodes against relative-pose constraints.

    Parameters
    ----------
    nodes:
        Ordered, non-empty iterable of ``(id, Pose3)`` initial poses. IDs are
        unique non-boolean integers; order is preserved in the result.
    constraints:
        Iterable of ``(from_id, to_id, measurement, information)`` tuples.
        ``measurement`` is the :class:`Pose3` for ``pose_from^-1 * pose_to``;
        ``information`` is the symmetric positive definite 6x6 weight matrix
        ordered like :meth:`Pose3.log` (rotation components first). Both
        endpoint IDs must appear in ``nodes``.
    fixed_node_ids:
        Non-empty iterable of node IDs held constant. Every connected
        component of the constraint graph must contain at least one fixed
        node; fixed poses are returned value-identical to their inputs.
    max_iterations:
        Maximum number of Gauss-Newton updates; a positive integer,
        default 50.
    tolerance:
        Convergence threshold on the total error; a finite positive number,
        default 1e-6. An initial total error not exceeding it returns
        immediately with ``iterations=0``; otherwise the loop stops once the
        absolute change of the total error between two consecutive updates is
        within tolerance.

    Returns
    -------
    PoseGraphResult
        ``poses`` in input node order, ``converged``, the number of performed
        ``iterations``, and ``initial_error``/``final_error`` recomputed from
        the input and returned poses. Exhausting ``max_iterations`` returns
        the last poses with ``converged=False``.
    """
    validated_nodes = _nodes(nodes)
    known_ids = {node_id for node_id, _ in validated_nodes}
    validated_constraints = _constraints(constraints, known_ids)
    fixed = _fixed_node_ids(fixed_node_ids, known_ids)
    iteration_limit = _positive_int(max_iterations, "max_iterations")
    convergence_tolerance = _positive_float(tolerance, "tolerance")
    _check_components_fixed(validated_nodes, validated_constraints, fixed)

    poses: dict[int, Pose3] = dict(validated_nodes)
    free_ids = tuple(node_id for node_id, _ in validated_nodes if node_id not in fixed)

    initial_error = _total_error(validated_constraints, poses)
    if initial_error <= convergence_tolerance:
        return PoseGraphResult(
            tuple(validated_nodes), True, 0, initial_error, initial_error
        )

    converged = False
    iterations = 0
    previous_error = initial_error
    current_error = initial_error
    for _ in range(iteration_limit):
        if free_ids:
            delta = _gauss_newton_step(validated_constraints, poses, free_ids)
            for slot, node_id in enumerate(free_ids):
                offset = 6 * slot
                poses[node_id] = Pose3.exp(delta[offset : offset + 6]).compose(
                    poses[node_id]
                )
        iterations += 1
        current_error = _total_error(validated_constraints, poses)
        if abs(current_error - previous_error) <= convergence_tolerance:
            converged = True
            break
        previous_error = current_error

    # Recompute the reported error from the final poses so it can never
    # describe a stale intermediate state.
    final_error = _total_error(validated_constraints, poses)
    result_poses = tuple((node_id, poses[node_id]) for node_id, _ in validated_nodes)
    return PoseGraphResult(
        result_poses, converged, iterations, initial_error, final_error
    )
