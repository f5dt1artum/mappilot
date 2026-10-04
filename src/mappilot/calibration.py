"""Sensor-to-body extrinsic calibration from paired relative motions.

The :func:`calibrate_extrinsic` entry point estimates the rigid extrinsic
``X`` (sensor coordinates -> body coordinates) from observations of motions
that a body and a sensor measured over the same time intervals. Because only
relative motions enter the residual, neither trajectory needs to share a
world origin with the other.

Every observation is ``(body_motion, sensor_motion, weight)`` with two
:class:`~mappilot.geometry.Pose3` motions ``B`` and ``S`` and a positive
finite weight ``w``. The residual of an extrinsic ``X`` is

    e = log( (B X)^{-1} (X S) )

(a 6D tangent vector ordered like :meth:`Pose3.log`:
``(wx, wy, wz, vx, vy, vz)``); the objective is the weighted sum

    F(X) = sum_k w_k * ||e_k||^2 .

It is minimized by Gauss-Newton iteration under the left-perturbation
convention used by :meth:`Pose3.exp`/:meth:`Pose3.log` (``X <- exp(dx) X``).
For one observation, writing ``D_k = Ad_{X^{-1} B^{-1}} - Ad_{X^{-1}}`` and
``J_l`` for the SE(3) left Jacobian,

    e_k = log( (B_k X)^{-1} (X S_k) )
    J_k = J_l(e_k)^{-1} D_k

so each normal-equation update solves ``H dx = -g`` with
``H = sum w_k J_k^T J_k`` and ``g = sum w_k J_k^T e_k``. Should a pure
Gauss-Newton step be singular or fail to reduce the objective, Levenberg-style
diagonal damping is added until a finite non-increasing step exists.

Observability: the stacked Jacobian is rank deficient exactly when the
motions cannot uniquely constrain all six extrinsic degrees of freedom. The
rank is independent of the current iterate because
``D_k(X) = Ad_{X^{-1}}(Ad_{B_k^{-1}} - I)`` only differs from its value at
the identity by an invertible common factor; it is therefore tested once up
front via the singular values of the weighted initial Jacobian. All-identity
motions constrain nothing; motions whose effective rotations are all zero,
or all share a single rotation axis (translations cannot make the missing
axis observable), are rejected as degenerate.

This module is pure Python and depends only on the standard library plus
:mod:`mappilot.geometry`. Importing it never starts the HTTP service.

Validation policy mirrors :mod:`mappilot.pose_graph`: a non-iterable
observations sequence, malformed observation records, motions or
``initial_pose`` that are not :class:`Pose3`, non-real (including boolean)
weights or control parameters raise :class:`TypeError`; fewer than three
observations, non-positive or non-finite weights, non-positive or
non-finite ``max_iterations``/``tolerance``, non-finite values produced
during optimization, and motion sets that cannot uniquely constrain the
6-DoF extrinsic raise :class:`ValueError`, and no partial result is
returned.
"""

from __future__ import annotations

import math
import numbers
from typing import Iterable, NamedTuple

from .geometry import Pose3

__all__ = ["ExtrinsicCalibrationResult", "calibrate_extrinsic"]

_DOF = 6

# Machine epsilon for the observability rank test.
_EPS = 2.220446049250313e-16

# Terms of the SE(3) left-Jacobian power series, matching pose_graph.py.
_JACOBIAN_TERMS = 24

# Levenberg damping schedule used when a plain Gauss-Newton step is not
# viable (singular factor or a step that fails to reduce the objective).
_DAMPING_START = 1e-3
_DAMPING_GROWTH = 10.0
_DAMPING_LIMIT = 1e12


class ExtrinsicCalibrationResult(NamedTuple):
    """Outcome of :func:`calibrate_extrinsic`.

    Immutable. ``sensor_to_body`` is the estimated extrinsic; ``residuals``
    holds the final six-dimensional residuals in exactly the order the
    observations were supplied. ``initial_error`` and ``final_error`` are
    recomputed from the corresponding poses and residuals, so every field is
    mutually consistent.
    """

    sensor_to_body: Pose3
    converged: bool
    iterations: int
    initial_error: float
    final_error: float
    residuals: tuple[tuple[float, float, float, float, float, float], ...]


# -- scalar / record validation ---------------------------------------------


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


def _weight(value: object, index: int) -> float:
    """Validate one positive finite observation weight."""
    name = f"observations[{index}][2]"
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {result!r}")
    if result <= 0.0:
        raise ValueError(f"{name} must be greater than zero, got {result}")
    return result


def _unpack_observation(
    record: object, index: int
) -> tuple[Pose3, Pose3, float]:
    """Validate one ``(body_motion, sensor_motion, weight)`` record."""
    name = f"observations[{index}]"
    if isinstance(record, (str, bytes, bytearray)):
        raise TypeError(
            f"{name} must be a (body_motion, sensor_motion, weight) triple"
        )
    try:
        items = tuple(record)  # type: ignore[call-overload]
    except TypeError:
        raise TypeError(
            f"{name} must be a (body_motion, sensor_motion, weight) triple"
        ) from None
    if len(items) != 3:
        raise TypeError(
            f"{name} must have exactly 3 elements "
            f"(body_motion, sensor_motion, weight), got {len(items)}"
        )
    body_motion, sensor_motion, raw_weight = items
    if not isinstance(body_motion, Pose3):
        raise TypeError(
            f"observations[{index}][0] must be a Pose3, got "
            f"{type(body_motion).__name__}"
        )
    if not isinstance(sensor_motion, Pose3):
        raise TypeError(
            f"observations[{index}][1] must be a Pose3, got "
            f"{type(sensor_motion).__name__}"
        )
    return body_motion, sensor_motion, _weight(raw_weight, index)


def _parse_observations(
    value: object,
) -> tuple[tuple[Pose3, Pose3, float], ...]:
    """Materialize the observation iterable exactly once and validate it."""
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(
            "observations must be an iterable of "
            "(body_motion, sensor_motion, weight) triples"
        )
    try:
        raw_records = list(value)  # type: ignore[call-overload]
    except TypeError:
        raise TypeError(
            "observations must be an iterable of "
            "(body_motion, sensor_motion, weight) triples"
        ) from None
    if len(raw_records) < 3:
        raise ValueError(
            "observations must contain at least three records, got "
            f"{len(raw_records)}"
        )
    return tuple(
        _unpack_observation(record, index)
        for index, record in enumerate(raw_records)
    )


# -- small dense linear algebra (standard library only) ----------------------


def _mat_mul(a, b):
    size = len(a)
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(size)) for j in range(size))
        for i in range(size)
    )


def _mat_sub(a, b):
    size = len(a)
    return tuple(
        tuple(a[i][j] - b[i][j] for j in range(size)) for i in range(size)
    )


def _identity(size: int):
    return tuple(
        tuple(1.0 if i == j else 0.0 for j in range(size)) for i in range(size)
    )


def _solve_linear(matrix, rhs):
    """Solve ``A x = b`` by Gaussian elimination with partial pivoting.

    Returns ``None`` when the matrix is singular to working precision.
    """
    size = len(matrix)
    augmented = [list(matrix[i]) + [rhs[i]] for i in range(size)]
    for column in range(size):
        pivot = column
        for row in range(column + 1, size):
            if abs(augmented[row][column]) > abs(augmented[pivot][column]):
                pivot = row
        pivot_value = augmented[pivot][column]
        if pivot_value == 0.0 or not math.isfinite(pivot_value):
            return None
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
        value = total / augmented[row][row]
        if not math.isfinite(value):
            return None
        solution[row] = value
    return solution


def _invert_matrix(matrix):
    """Invert a non-singular square matrix by solving against identity cols."""
    size = len(matrix)
    columns = []
    for basis in range(size):
        rhs = [0.0] * size
        rhs[basis] = 1.0
        solved = _solve_linear(matrix, rhs)
        if solved is None:
            raise ValueError("linear system is singular")
        columns.append(solved)
    return tuple(
        tuple(columns[j][i] for j in range(size)) for i in range(size)
    )


def _column_rank(rows, observation_count: int) -> int:
    """Column rank of ``rows`` by pivoted modified Gram-Schmidt QR.

    Working on the stacked Jacobian directly (instead of the squared normal
    matrix) keeps the numerical null threshold at machine epsilon rather
    than its square root.
    """
    if not rows:
        return 0
    dimension = len(rows[0])
    row_count = len(rows)
    columns = [[row[column] for row in rows] for column in range(dimension)]
    largest = max(math.sqrt(sum(value * value for value in column))
                  for column in columns)
    if largest == 0.0:
        return 0
    threshold = largest * max(3, observation_count) * _EPS
    rank = 0
    for step in range(dimension):
        pivot = max(
            range(step, dimension),
            key=lambda j: math.sqrt(sum(value * value for value in columns[j])),
        )
        pivot_norm = math.sqrt(sum(value * value for value in columns[pivot]))
        if pivot_norm <= threshold:
            return rank
        columns[step], columns[pivot] = columns[pivot], columns[step]
        pivot_column = columns[step]
        unit = [value / pivot_norm for value in pivot_column]
        for j in range(step + 1, dimension):
            projection = sum(
                unit[k] * columns[j][k] for k in range(row_count)
            )
            if projection != 0.0:
                columns[j] = [
                    columns[j][k] - projection * unit[k] for k in range(row_count)
                ]
        rank += 1
    return rank


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
    """6x6 matrix ``ad_xi`` for ``xi = (wx, wy, wz, vx, vy, vz)``."""
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
    """``J_l(xi)^{-1}`` as a 6x6 matrix, via the left-Jacobian series."""
    ad = _se3_adjoint_algebra(xi)
    identity = _identity(_DOF)
    series = identity
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


# -- residual / objective -----------------------------------------------------


def _residual_and_jacobian(
    extrinsic: Pose3, body_motion: Pose3, sensor_motion: Pose3
):
    """Return ``(e, J)`` for ``e = log((B X)^{-1} (X S))``.

    For the left perturbation ``X <- exp(d) X`` the differential of the
    residual is

        J = J_l(e)^{-1} ( Ad_{X^{-1} B^{-1}} - Ad_{X^{-1}} )
    """
    error_pose = body_motion.compose(extrinsic).inverse().compose(
        extrinsic.compose(sensor_motion)
    )
    residual = error_pose.log()
    extrinsic_inverse = extrinsic.inverse()
    delta = _mat_sub(
        _adjoint(extrinsic_inverse.compose(body_motion.inverse())),
        _adjoint(extrinsic_inverse),
    )
    jacobian = _mat_mul(_inverse_left_jacobian(residual), delta)
    return residual, jacobian


def _residual(extrinsic: Pose3, body_motion: Pose3, sensor_motion: Pose3):
    """Return just ``e = log((B X)^{-1} (X S))`` for error evaluation."""
    error_pose = body_motion.compose(extrinsic).inverse().compose(
        extrinsic.compose(sensor_motion)
    )
    return error_pose.log()


def _linearize(extrinsic: Pose3, records, with_jacobians: bool):
    """Evaluate residuals and optionally the Gauss-Newton normal equations.

    Raises :class:`ValueError` if any computed value is non-finite.
    """
    residuals: list[tuple[float, ...]] = []
    error = 0.0
    hessian = [[0.0] * _DOF for _ in range(_DOF)]
    gradient = [0.0] * _DOF
    for body_motion, sensor_motion, weight in records:
        if with_jacobians:
            residual, jacobian = _residual_and_jacobian(
                extrinsic, body_motion, sensor_motion
            )
            for row in jacobian:
                for entry in row:
                    if not math.isfinite(entry):
                        raise ValueError(
                            "extrinsic calibration produced non-finite Jacobian"
                        )
        else:
            residual = _residual(extrinsic, body_motion, sensor_motion)
        for component in residual:
            if not math.isfinite(component):
                raise ValueError("extrinsic calibration produced non-finite residual")
        residuals.append(residual)
        squared = sum(component * component for component in residual)
        error += weight * squared
        if with_jacobians:
            for row in range(_DOF):
                gradient[row] += weight * sum(
                    jacobian[k][row] * residual[k] for k in range(_DOF)
                )
                target = hessian[row]
                for col in range(_DOF):
                    target[col] += weight * sum(
                        jacobian[k][row] * jacobian[k][col] for k in range(_DOF)
                    )
    if not math.isfinite(error):
        raise ValueError("extrinsic calibration produced non-finite error")
    if with_jacobians:
        for value in (*gradient, *(value for row in hessian for value in row)):
            if not math.isfinite(value):
                raise ValueError(
                    "extrinsic calibration produced non-finite normal equations"
                )
    return error, tuple(residuals), hessian, gradient


def _initial_error_and_rank(extrinsic: Pose3, records):
    """Evaluate the initial objective and the stacked-Jacobian column rank.

    The Jacobian rows are scaled by ``sqrt(weight)`` so the rank test sees
    exactly the metric defining the weighted least-squares problem.
    """
    residuals: list[tuple[float, ...]] = []
    error = 0.0
    rows: list[tuple[float, ...]] = []
    for body_motion, sensor_motion, weight in records:
        residual, jacobian = _residual_and_jacobian(
            extrinsic, body_motion, sensor_motion
        )
        for component in residual:
            if not math.isfinite(component):
                raise ValueError("extrinsic calibration produced non-finite residual")
        for row in jacobian:
            for entry in row:
                if not math.isfinite(entry):
                    raise ValueError("extrinsic calibration produced non-finite Jacobian")
        residuals.append(residual)
        error += weight * sum(component * component for component in residual)
        scale = math.sqrt(weight)
        for row in jacobian:
            rows.append(tuple(scale * entry for entry in row))
    if not math.isfinite(error):
        raise ValueError("extrinsic calibration produced non-finite error")
    return error, tuple(residuals), _column_rank(rows, len(records))


# -- public entry point -------------------------------------------------------


def calibrate_extrinsic(
    observations: Iterable[tuple[Pose3, Pose3, float]],
    initial_pose: Pose3 | None = None,
    max_iterations: int = 50,
    tolerance: float = 1e-6,
) -> ExtrinsicCalibrationResult:
    """Estimate the sensor-to-body extrinsic from paired relative motions.

    Parameters
    ----------
    observations:
        Iterable of ``(body_motion, sensor_motion, weight)`` triples; both
        motions are :class:`~mappilot.geometry.Pose3` relative motions
        measured over the same interval and the weight is a positive finite
        real number. At least three records are required. The iterable is
        consumed exactly once.
    initial_pose:
        Initial guess for the sensor-to-body extrinsic; ``None`` (default)
        uses the identity pose.
    max_iterations:
        Maximum number of updates; a positive integer, default 50.
    tolerance:
        Finite positive convergence threshold on the absolute change in the
        weighted objective between consecutive updates, default 1e-6.

    Returns
    -------
    ExtrinsicCalibrationResult
        If the initial objective is already at most ``tolerance``, the result
        reports ``converged=True`` with ``iterations=0`` and the initial pose
        untouched. Otherwise the loop stops as soon as the absolute objective
        change between two updates is at most ``tolerance``
        (``converged=True``), or after ``max_iterations`` updates with
        ``converged=False``; exhaustion never raises. ``residuals`` are in
        input order and ``initial_error``/``final_error`` are recomputed from
        the corresponding poses.

    Raises
    ------
    TypeError
        A non-iterable observations sequence, malformed observation records,
        motions or ``initial_pose`` that are not :class:`Pose3`, or
        weights/control parameters of the wrong numeric type.
    ValueError
        Fewer than three observations; non-positive or non-finite weights;
        non-positive or non-finite ``max_iterations``/``tolerance``;
        non-finite values produced during optimization; or motion sets that
        cannot uniquely constrain the 6-DoF extrinsic (all-identity motions,
        zero effective rotation, or a single rotation axis that
        translations cannot complement).
    """
    records = _parse_observations(observations)
    if initial_pose is None:
        extrinsic = Pose3.identity()
    elif isinstance(initial_pose, Pose3):
        extrinsic = initial_pose
    else:
        raise TypeError(
            "initial_pose must be a Pose3 or None, got "
            f"{type(initial_pose).__name__}"
        )
    iteration_limit = _positive_int(max_iterations, "max_iterations")
    convergence_tolerance = _positive_float(tolerance, "tolerance")

    initial_error, initial_residuals, rank = _initial_error_and_rank(
        extrinsic, records
    )
    if rank < _DOF:
        raise ValueError(
            "observations cannot uniquely constrain the 6-DoF extrinsic: "
            "the motion set is degenerate (all-identity motions, zero "
            "effective rotation, or only a single rotation axis)"
        )

    if initial_error <= convergence_tolerance:
        return ExtrinsicCalibrationResult(
            extrinsic,
            True,
            0,
            initial_error,
            initial_error,
            initial_residuals,
        )

    converged = False
    iterations = 0
    previous_error = initial_error

    for _ in range(iteration_limit):
        error_at_x, _, hessian, gradient = _linearize(
            extrinsic, records, with_jacobians=True
        )
        # ``error_at_x`` equals the accepted error of the previous update.
        previous_error = error_at_x

        neg_gradient = tuple(-value for value in gradient)
        diagonal = tuple(hessian[i][i] for i in range(_DOF))
        damping = 0.0
        candidate = None
        candidate_error = math.inf

        # Plain Gauss-Newton first; add Levenberg damping only when the
        # factor is singular or the step fails to be non-increasing.
        while True:
            if damping == 0.0:
                damped = [row[:] for row in hessian]
            else:
                damped = [
                    [
                        hessian[i][j] + (damping * diagonal[i] if i == j else 0.0)
                        for j in range(_DOF)
                    ]
                    for i in range(_DOF)
                ]
            solved = _solve_linear(damped, neg_gradient)
            if solved is not None:
                try:
                    proposed = Pose3.exp(tuple(solved)).compose(extrinsic)
                    proposed_error, _, _, _ = _linearize(
                        proposed, records, with_jacobians=False
                    )
                except (OverflowError, ValueError):
                    proposed_error = math.inf
                else:
                    if (
                        math.isfinite(proposed_error)
                        and proposed_error
                        <= previous_error
                        + 1e-14 * max(1.0, abs(previous_error))
                    ):
                        step = solved
                        candidate = proposed
                        candidate_error = proposed_error
            if candidate is not None:
                break
            if damping == 0.0:
                damping = _DAMPING_START
            else:
                damping *= _DAMPING_GROWTH
            if damping > _DAMPING_LIMIT:
                raise ValueError(
                    "extrinsic calibration failed to produce a finite "
                    "objective-reducing step"
                )

        extrinsic = candidate
        iterations += 1

        if abs(previous_error - candidate_error) <= convergence_tolerance:
            converged = True
            break
        previous_error = candidate_error

    final_error, final_residuals, _, _ = _linearize(
        extrinsic, records, with_jacobians=False
    )
    return ExtrinsicCalibrationResult(
        extrinsic,
        converged,
        iterations,
        initial_error,
        final_error,
        final_residuals,
    )
