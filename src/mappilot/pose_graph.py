"""Pose-graph optimization over SE(3) relative-pose constraints.

The :func:`optimize_pose_graph` entry point takes an ordered, uniquely
identified sequence of :class:`~mappilot.geometry.Pose3` node initial values
and a sequence of relative-pose constraints, and produces a globally
consistent trajectory by Gauss-Newton iteration.

For every constraint ``(i, j, z, Omega)`` the residual is

    e = log( z^{-1} * T_i^{-1} * T_j )

(a 6D tangent vector ordered like :meth:`Pose3.log`:
``(wx, wy, wz, vx, vy, vz)``) and the total error is the sum of quadratic
forms ``e^T Omega e``. ``Omega`` must be a symmetric positive-definite 6x6
matrix whose entries align with the :meth:`Pose3.log` components.

Each Gauss-Newton step linearizes the residual under the left-perturbation
convention used by :meth:`Pose3.exp`/:meth:`Pose3.log` (``T <- exp(dx) T``):

    J_j =  J_l(e)^{-1} Ad_{z^{-1} T_i^{-1}}
    J_i = -J_j

solves the reduced normal equations (rows and columns of fixed nodes
eliminated) by pivoted Gaussian elimination, and applies the update
simultaneously to every free node. Fixed node poses are never touched, so
disconnected components are optimized independently as long as each
component carries at least one fixed node.

This module is pure Python and depends only on the standard library plus
:mod:`mappilot.geometry`. Importing it never starts the HTTP service.

Validation policy mirrors :mod:`mappilot.geometry` and
:mod:`mappilot.registration`: non-iterable inputs, non-integer (including
boolean) node ids, non-:class:`Pose3` poses, malformed constraint tuples or
information matrices, non-real (including boolean) matrix entries, and
parameters of the wrong type raise :class:`TypeError`; empty or duplicate
node ids, constraints referencing unknown nodes, unknown or empty fixed
sets, connected components without a fixed node, non-finite matrix
entries, information matrices that are not symmetric (absolute tolerance
1e-12) or not positive definite, and non-positive or non-finite
``max_iterations``/``tolerance`` raise :class:`ValueError`.
"""

from __future__ import annotations

import math
import numbers
from typing import Iterable, NamedTuple

from .geometry import Pose3

__all__ = ["PoseGraphResult", "optimize_pose_graph"]

_DOF = 6

# Information-matrix symmetry tolerance.
_SYMMETRY_TOL = 1e-12

# Terms of the SE(3) left-Jacobian power series (ad_x^n / (n+1)!). With the
# log angle in [0, pi] the tail past n=24 is below ~1e-12 in exact
# arithmetic; 24 leaves a wide safety margin in double precision.
_JACOBIAN_TERMS = 24


class PoseGraphResult(NamedTuple):
    """Outcome of :func:`optimize_pose_graph`.

    Immutable. ``nodes`` holds ``(id, pose)`` pairs in exactly the order the
    nodes were supplied; fixed node poses are the input objects, unchanged
    value by value. ``initial_error`` and ``final_error`` are recomputed from
    the corresponding poses, so they always agree with ``nodes``.
    """

    nodes: tuple[tuple[int, Pose3], ...]
    converged: bool
    iterations: int
    initial_error: float
    final_error: float


# -- scalar / sequence validation -------------------------------------------


def _node_id(value: object, name: str) -> int:
    """Coerce one non-boolean integral node id to int."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer node id, got {type(value).__name__}")
    return int(value)


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


def _matrix_entry(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {result!r}")
    return result


def _pair(value: object, name: str, length: int) -> tuple:
    """Materialize one fixed-length input tuple; type-error on bad shape."""
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be a sequence of {length} elements")
    try:
        items = tuple(value)  # type: ignore[call-overload]
    except TypeError:
        raise TypeError(f"{name} must be a sequence of {length} elements") from None
    if len(items) != length:
        raise TypeError(f"{name} must have exactly {length} elements, got {len(items)}")
    return items


# -- information matrix ------------------------------------------------------


def _information_matrix(value: object) -> tuple[tuple[float, ...], ...]:
    """Validate a symmetric positive-definite 6x6 real matrix."""
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError("information must be a 6x6 nested sequence of real numbers")
    try:
        rows = tuple(value)  # type: ignore[call-overload]
    except TypeError:
        raise TypeError("information must be a 6x6 nested sequence of real numbers") from None
    if len(rows) != _DOF:
        raise TypeError(f"information must have {_DOF} rows, got {len(rows)}")

    grid: list[tuple[float, ...]] = []
    for row_index, row in enumerate(rows):
        if isinstance(row, (str, bytes, bytearray)):
            raise TypeError(f"information[{row_index}] must be a sequence of 6 real numbers")
        try:
            items = tuple(row)  # type: ignore[call-overload]
        except TypeError:
            raise TypeError(
                f"information[{row_index}] must be a sequence of 6 real numbers"
            ) from None
        if len(items) != _DOF:
            raise TypeError(
                f"information[{row_index}] must have {_DOF} elements, got {len(items)}"
            )
        grid.append(
            tuple(
                _matrix_entry(item, f"information[{row_index}][{col_index}]")
                for col_index, item in enumerate(items)
            )
        )

    matrix = tuple(grid)
    for i in range(_DOF):
        for j in range(i + 1, _DOF):
            if abs(matrix[i][j] - matrix[j][i]) > _SYMMETRY_TOL:
                raise ValueError("information matrix must be symmetric")

    # Positive definiteness via symmetric Gaussian elimination (LDL^T without
    # pivoting): a symmetric matrix is positive definite iff every pivot is
    # positive. A symmetric positive-definite matrix never needs pivoting.
    work = [list(row) for row in matrix]
    for k in range(_DOF):
        pivot = work[k][k]
        if pivot <= 0.0:
            raise ValueError("information matrix must be positive definite")
        for i in range(k + 1, _DOF):
            factor = work[i][k] / pivot
            if factor == 0.0:
                continue
            for j in range(k + 1, _DOF):
                work[i][j] -= factor * work[k][j]
    return matrix


# -- nodes / constraints / fixed nodes ---------------------------------------


def _parse_nodes(
    value: object,
) -> tuple[tuple[tuple[int, Pose3], ...], dict[int, int]]:
    """Validate the ordered node sequence; return pairs and id->index."""
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError("nodes must be an iterable of (id, pose) pairs")
    try:
        raw_nodes = tuple(value)  # type: ignore[call-overload]
    except TypeError:
        raise TypeError("nodes must be an iterable of (id, pose) pairs") from None
    if len(raw_nodes) == 0:
        raise ValueError("nodes must not be empty")

    nodes: list[tuple[int, Pose3]] = []
    index_by_id: dict[int, int] = {}
    for position, raw_node in enumerate(raw_nodes):
        items = _pair(raw_node, f"nodes[{position}]", 2)
        node_id = _node_id(items[0], f"nodes[{position}][0]")
        pose = items[1]
        if not isinstance(pose, Pose3):
            raise TypeError(
                f"nodes[{position}][1] must be a Pose3, got {type(pose).__name__}"
            )
        if node_id in index_by_id:
            raise ValueError(f"duplicate node id: {node_id}")
        index_by_id[node_id] = position
        nodes.append((node_id, pose))
    return tuple(nodes), index_by_id


def _parse_constraints(
    value: object, index_by_id: dict[int, int]
) -> tuple[tuple[int, int, Pose3, tuple[tuple[float, ...], ...]], ...]:
    """Validate ``(from_id, to_id, measurement, information)`` constraints."""
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(
            "constraints must be an iterable of "
            "(from_id, to_id, measurement, information) tuples"
        )
    try:
        raw_constraints = tuple(value)  # type: ignore[call-overload]
    except TypeError:
        raise TypeError(
            "constraints must be an iterable of "
            "(from_id, to_id, measurement, information) tuples"
        ) from None

    constraints: list[tuple[int, int, Pose3, tuple[tuple[float, ...], ...]]] = []
    for position, raw_constraint in enumerate(raw_constraints):
        name = f"constraints[{position}]"
        items = _pair(
            raw_constraint,
            name,
            4,
        )
        from_id = _node_id(items[0], f"{name}[0]")
        to_id = _node_id(items[1], f"{name}[1]")
        measurement = items[2]
        if not isinstance(measurement, Pose3):
            raise TypeError(
                f"{name}[2] must be a Pose3, got {type(measurement).__name__}"
            )
        information = _information_matrix(items[3])
        if from_id not in index_by_id:
            raise ValueError(f"{name} references unknown node id: {from_id}")
        if to_id not in index_by_id:
            raise ValueError(f"{name} references unknown node id: {to_id}")
        constraints.append(
            (index_by_id[from_id], index_by_id[to_id], measurement, information)
        )
    return tuple(constraints)


def _parse_fixed(value: object, index_by_id: dict[int, int]) -> frozenset[int]:
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError("fixed_node_ids must be an iterable of integer node ids")
    try:
        raw_ids = tuple(value)  # type: ignore[call-overload]
    except TypeError:
        raise TypeError("fixed_node_ids must be an iterable of integer node ids") from None
    fixed: set[int] = set()
    for position, raw_id in enumerate(raw_ids):
        node_id = _node_id(raw_id, f"fixed_node_ids[{position}]")
        if node_id not in index_by_id:
            raise ValueError(f"fixed node id is unknown: {node_id}")
        fixed.add(index_by_id[node_id])
    if not fixed:
        raise ValueError("fixed_node_ids must not be empty")
    return frozenset(fixed)


def _check_components(
    node_count: int,
    constraints: tuple[tuple[int, int, Pose3, tuple[tuple[float, ...], ...]], ...],
    fixed: frozenset[int],
) -> None:
    """Raise if any undirected connected component lacks a fixed node."""
    adjacency: list[list[int]] = [[] for _ in range(node_count)]
    for from_index, to_index, _, _ in constraints:
        adjacency[from_index].append(to_index)
        adjacency[to_index].append(from_index)

    seen = [False] * node_count
    for start in range(node_count):
        if seen[start]:
            continue
        seen[start] = True
        stack = [start]
        component_has_fixed = start in fixed
        while stack:
            node = stack.pop()
            for neighbour in adjacency[node]:
                if not seen[neighbour]:
                    seen[neighbour] = True
                    if neighbour in fixed:
                        component_has_fixed = True
                    stack.append(neighbour)
        if not component_has_fixed:
            raise ValueError(
                "every connected component must contain at least one fixed node; "
                f"the component containing node index {start} has none"
            )


# -- small dense linear algebra (standard library only) ----------------------


def _mat_mul(a, b):
    """Product of two square matrices stored as tuples of tuples."""
    size = len(a)
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(size)) for j in range(size))
        for i in range(size)
    )


def _mat_vec_mul(matrix, vector):
    size = len(matrix)
    return tuple(
        sum(matrix[i][k] * vector[k] for k in range(size)) for i in range(size)
    )


def _identity(size: int):
    return tuple(
        tuple(1.0 if i == j else 0.0 for j in range(size)) for i in range(size)
    )


def _solve_linear(matrix, rhs):
    """Solve ``A x = b`` by Gaussian elimination with partial pivoting.

    Returns a list. Raises :class:`ValueError` if the system is singular to
    working precision.
    """
    size = len(matrix)
    augmented = [list(matrix[i]) + [rhs[i]] for i in range(size)]
    for column in range(size):
        pivot = column
        for row in range(column + 1, size):
            if abs(augmented[row][column]) > abs(augmented[pivot][column]):
                pivot = row
        if augmented[pivot][column] == 0.0 or not math.isfinite(
            augmented[pivot][column]
        ):
            raise ValueError("linear system is singular")
        if pivot != column:
            augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        pivot_value = augmented[column][column]
        for row in range(column + 1, size):
            factor = augmented[row][column] / pivot_value
            if factor == 0.0:
                continue
            for col in range(column, size + 1):
                augmented[row][col] -= factor * augmented[column][col]
    solution = [0.0] * size
    for row in range(size - 1, -1, -1):
        total = augmented[row][size]
        for col in range(row + 1, size):
            total -= augmented[row][col] * solution[col]
        solution[row] = total / augmented[row][row]
    return solution


def _invert_matrix(matrix):
    """Invert a non-singular square matrix by solving against identity cols."""
    size = len(matrix)
    columns = []
    for basis in range(size):
        rhs = [0.0] * size
        rhs[basis] = 1.0
        columns.append(_solve_linear(matrix, rhs))
    return tuple(
        tuple(columns[j][i] for j in range(size)) for i in range(size)
    )


# -- SE(3) adjoint and left Jacobian -----------------------------------------


def _skew3(x: float, y: float, z: float):
    return (
        (0.0, -z, y),
        (z, 0.0, -x),
        (-y, x, 0.0),
    )


def _adjoint(pose: Pose3):
    """``Ad_T = [[R, 0], [[t]x R, R]]`` in tangent order (omega, v)."""
    rotation = pose.rotation
    tx, ty, tz = pose.translation
    skew_t = _skew3(tx, ty, tz)
    bottom = tuple(
        tuple(
            sum(skew_t[row][k] * rotation[k][col] for k in range(3))
            for col in range(3)
        )
        for row in range(3)
    )
    return (
        (rotation[0][0], rotation[0][1], rotation[0][2], 0.0, 0.0, 0.0),
        (rotation[1][0], rotation[1][1], rotation[1][2], 0.0, 0.0, 0.0),
        (rotation[2][0], rotation[2][1], rotation[2][2], 0.0, 0.0, 0.0),
        (bottom[0][0], bottom[0][1], bottom[0][2],
         rotation[0][0], rotation[0][1], rotation[0][2]),
        (bottom[1][0], bottom[1][1], bottom[1][2],
         rotation[1][0], rotation[1][1], rotation[1][2]),
        (bottom[2][0], bottom[2][1], bottom[2][2],
         rotation[2][0], rotation[2][1], rotation[2][2]),
    )


def _se3_adjoint_algebra(xi):
    """6x6 matrix ``ad_xi`` for ``xi = (wx, wy, wz, vx, vy, vz)``.

    Block form ``[[w]x, 0; [v]x, [w]x]``; the left-Jacobian series
    ``sum ad_xi^n / (n+1)!`` is built from this matrix.
    """
    wx, wy, wz, vx, vy, vz = xi
    kw = _skew3(wx, wy, wz)
    kv = _skew3(vx, vy, vz)
    return (
        (kw[0][0], kw[0][1], kw[0][2], 0.0, 0.0, 0.0),
        (kw[1][0], kw[1][1], kw[1][2], 0.0, 0.0, 0.0),
        (kw[2][0], kw[2][1], kw[2][2], 0.0, 0.0, 0.0),
        (kv[0][0], kv[0][1], kv[0][2], kw[0][0], kw[0][1], kw[0][2]),
        (kv[1][0], kv[1][1], kv[1][2], kw[1][0], kw[1][1], kw[1][2]),
        (kv[2][0], kv[2][1], kv[2][2], kw[2][0], kw[2][1], kw[2][2]),
    )


def _inverse_left_jacobian(xi):
    """``J_l(xi)^{-1}`` as a 6x6 matrix.

    ``J_l(xi) = sum_{n>=0} ad_xi^n / (n+1)!`` evaluated by Horner's rule,
    then inverted by pivoted Gaussian elimination. :meth:`Pose3.log` bounds
    the rotation angle by pi, well inside the series' convergence radius of
    2*pi, so the truncated tail is negligible.
    """
    ad = _se3_adjoint_algebra(xi)
    identity = _identity(_DOF)
    series = identity
    # J = I + ad/2 (I + ad/3 (I + ... + ad/(TERMS+1) I ...))
    for divisor in range(_JACOBIAN_TERMS + 1, 1, -1):
        scaled = tuple(
            tuple(ad[i][j] / divisor for j in range(_DOF)) for i in range(_DOF)
        )
        series = tuple(
            tuple(
                identity[i][j] + sum(scaled[i][k] * series[k][j] for k in range(_DOF))
                for j in range(_DOF)
            )
            for i in range(_DOF)
        )
    return _invert_matrix(series)


# -- residual / error ---------------------------------------------------------


def _residual_and_jacobians(
    pose_from: Pose3, pose_to: Pose3, measurement: Pose3
):
    """Return ``(e, J_from, J_to)`` for one constraint ``log(z^-1 Ti^-1 Tj)``.

    Left perturbation ``T <- exp(d) T`` gives (see module docstring)::

        J_to   =  J_l(e)^{-1} Ad_{z^{-1} T_i^{-1}}
        J_from = -J_to
    """
    error_pose = measurement.inverse().compose(
        pose_from.inverse().compose(pose_to)
    )
    residual = error_pose.log()
    base = _mat_mul(
        _inverse_left_jacobian(residual),
        _adjoint(measurement.inverse().compose(pose_from.inverse())),
    )
    j_from = tuple(tuple(-value for value in row) for row in base)
    return residual, j_from, base


def _quadratic_form(information, residual) -> float:
    total = 0.0
    for i in range(_DOF):
        weighted = sum(information[i][k] * residual[k] for k in range(_DOF))
        total += residual[i] * weighted
    return total


def _total_error(poses, constraints) -> float:
    total = 0.0
    for from_index, to_index, measurement, information in constraints:
        residual = measurement.inverse().compose(
            poses[from_index].inverse().compose(poses[to_index])
        ).log()
        total += _quadratic_form(information, residual)
    return total


# -- public entry point -------------------------------------------------------


def optimize_pose_graph(
    nodes: Iterable[tuple[int, Pose3]],
    constraints: Iterable[tuple[int, Pose3, Pose3, tuple[tuple[float, ...], ...]]],
    fixed_node_ids: Iterable[int],
    max_iterations: int = 50,
    tolerance: float = 1e-6,
) -> PoseGraphResult:
    """Optimize a pose graph of :class:`~mappilot.geometry.Pose3` nodes.

    Parameters
    ----------
    nodes:
        Ordered sequence of unique ``(id, pose)`` pairs; ``id`` is a
        non-boolean integer and ``pose`` a :class:`~mappilot.geometry.Pose3`
        used as the Gauss-Newton initial value.
    constraints:
        Sequence of ``(from_id, to_id, measurement, information)`` tuples.
        ``measurement`` is the measured relative pose from the ``from`` node
        to the ``to`` node; ``information`` is a symmetric positive-definite
        6x6 matrix aligned with :meth:`Pose3.log` components.
    fixed_node_ids:
        Non-empty iterable of node ids held exactly at their input poses.
        Every connected component must contain at least one fixed node.
    max_iterations:
        Maximum number of Gauss-Newton updates; positive integer, default 50.
    tolerance:
        Finite positive convergence threshold on the absolute change in total
        error between consecutive updates, default 1e-6.

    Returns
    -------
    PoseGraphResult
        If the initial total error is already at most ``tolerance``, the
        result reports ``converged=True`` with ``iterations=0`` and the input
        poses untouched. Otherwise the loop stops as soon as the absolute
        error change between two updates is at most ``tolerance``
        (``converged=True``), or after ``max_iterations`` updates with
        ``converged=False``. ``initial_error`` and ``final_error`` are
        recomputed from the corresponding poses; fixed node poses compare
        equal to the inputs value by value.
    """
    parsed_nodes, index_by_id = _parse_nodes(nodes)
    parsed_constraints = _parse_constraints(constraints, index_by_id)
    fixed_indices = _parse_fixed(fixed_node_ids, index_by_id)
    iteration_limit = _positive_int(max_iterations, "max_iterations")
    convergence_tolerance = _positive_float(tolerance, "tolerance")
    _check_components(len(parsed_nodes), parsed_constraints, fixed_indices)

    node_ids = tuple(node_id for node_id, _ in parsed_nodes)
    poses: list[Pose3] = [pose for _, pose in parsed_nodes]

    initial_error = _total_error(poses, parsed_constraints)
    if initial_error <= tolerance:
        return PoseGraphResult(
            tuple(zip(node_ids, poses)),
            True,
            0,
            initial_error,
            _total_error(poses, parsed_constraints),
        )

    free_nodes = [i for i in range(len(poses)) if i not in fixed_indices]
    free_slot = {node_index: slot for slot, node_index in enumerate(free_nodes)}
    free_count = len(free_nodes)

    converged = False
    iterations = 0
    previous_error = initial_error

    for _ in range(iteration_limit):
        hessian = [[0.0] * (free_count * _DOF) for _ in range(free_count * _DOF)]
        gradient = [0.0] * (free_count * _DOF)

        for from_index, to_index, measurement, information in parsed_constraints:
            residual, j_from, j_to = _residual_and_jacobians(
                poses[from_index], poses[to_index], measurement
            )
            weighted_error = _mat_vec_mul(information, residual)

            for node_index, jacobian in (
                (from_index, j_from),
                (to_index, j_to),
            ):
                if node_index in fixed_indices:
                    continue
                slot = free_slot[node_index]
                for row in range(_DOF):
                    gradient[slot * _DOF + row] -= sum(
                        jacobian[k][row] * weighted_error[k] for k in range(_DOF)
                    )

            blocks = ((from_index, j_from), (to_index, j_to))
            for node_a, jac_a in blocks:
                if node_a in fixed_indices:
                    continue
                slot_a = free_slot[node_a]
                for node_b, jac_b in blocks:
                    if node_b in fixed_indices:
                        continue
                    slot_b = free_slot[node_b]
                    for row in range(_DOF):
                        target = hessian[slot_a * _DOF + row]
                        base = slot_b * _DOF
                        for col in range(_DOF):
                            target[base + col] += sum(
                                jac_a[k][row] * information[k][k2] * jac_b[k2][col]
                                for k in range(_DOF) for k2 in range(_DOF)
                            )

        # Gauss-Newton step: H dx = -J^T Omega e; apply T <- exp(dx) T.
        step = _solve_linear(hessian, gradient)
        for slot, node_index in enumerate(free_nodes):
            delta = tuple(step[slot * _DOF + k] for k in range(_DOF))
            poses[node_index] = Pose3.exp(delta).compose(poses[node_index])
        iterations += 1

        current_error = _total_error(poses, parsed_constraints)
        if abs(previous_error - current_error) <= convergence_tolerance:
            converged = True
            previous_error = current_error
            break
        previous_error = current_error

    return PoseGraphResult(
        tuple(zip(node_ids, poses)),
        converged,
        iterations,
        initial_error,
        _total_error(poses, parsed_constraints),
    )
