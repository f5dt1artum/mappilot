"""Extrinsic sensor-to-body calibration from paired relative motions.

:func:`calibrate_extrinsic` estimates the sensor-to-body
:class:`~mappilot.geometry.Pose3` ``X`` from paired relative motions. Each
observation supplies the body motion ``A`` and the sensor motion ``B``
over the same interval, each expressed in its own reference frame, so the
two motion sources never need to share a world origin. For the true
extrinsic the motions satisfy the hand-eye relation ``A X = X B``; the
residual of one observation is the 6D tangent vector

    e = log( (A X)^{-1} (X B) )

ordered like :meth:`Pose3.log` (``(wx, wy, wz, vx, vy, vz)``), and the
objective is the weighted sum ``sum(w * e . e)`` over all observations.

The estimate is refined by Gauss-Newton iteration under the
left-perturbation convention of :meth:`Pose3.exp`/:meth:`Pose3.log`
(``X <- exp(d) X``). Linearizing the residual about the current estimate
gives the exact 6x6 Jacobian

    J = J_l(e)^{-1} ( Ad_{(A X)^{-1}} - Ad_{X^{-1}} )

where ``J_l`` is the SE(3) left Jacobian and ``Ad`` the adjoint, both
shared with :mod:`mappilot.pose_graph` conventions. Each step solves the
weighted normal equations by pivoted Gaussian elimination and applies the
update simultaneously.

Observability: the stacked Jacobians have full column rank six exactly
when the rotation axes of the body motions with non-zero rotation span at
least two distinct directions. All-identity motions leave every residual
zero; rotations about a single axis leave the translation of ``X`` along
that axis (and, without translational coupling, the rotation about it)
unobservable, and pure translations can never constrain the translation
of ``X`` at all. Such motion sets are rejected as degenerate instead of
returning an arbitrary solution. Repeated motions are harmless: they add
no rank but are valid observations.

This module is pure Python and depends only on the standard library plus
:mod:`mappilot.geometry`. Importing it never starts the HTTP service.

Validation policy mirrors :mod:`mappilot.pose_graph` and
:mod:`mappilot.fusion`: non-iterable observations, malformed records,
motions or an ``initial_pose`` that are not :class:`Pose3`, and weights or
control parameters of the wrong type (including booleans) raise
:class:`TypeError`; fewer than three observations, non-positive or
non-finite weights, out-of-range control parameters, non-finite
intermediate results, and motion sets that cannot uniquely constrain the
six-degree-of-freedom extrinsic raise :class:`ValueError`.
"""

from __future__ import annotations

import math
import numbers
from typing import Iterable, NamedTuple

from .geometry import Pose3

__all__ = ["ExtrinsicCalibrationResult", "calibrate_extrinsic"]

_DOF = 6

# Terms of the SE(3) left-Jacobian power series (ad_x^n / (n+1)!). With the
# log angle in [0, pi] the tail past n=24 is below ~1e-12 in exact
# arithmetic; 24 leaves a wide safety margin in double precision.
_JACOBIAN_TERMS = 24

# A body motion whose rotation angle (radians) is at most this threshold is
# treated as rotation-free, and two unit axes whose cross-product magnitude
# is at most this threshold are treated as parallel, when deciding whether
# the motion set observably constrains all six extrinsic degrees of freedom.
_AXIS_EPSILON = 1e-9

# Minimum number of paired-motion observations.
_MIN_OBSERVATIONS = 3


class ExtrinsicCalibrationResult(NamedTuple):
    """Outcome of :func:`calibrate_extrinsic`.

    Immutable. ``sensor_to_body`` is the estimated extrinsic
    :class:`~mappilot.geometry.Pose3` ``X``. ``residuals`` holds the 6D
    residual tangent vector of every observation at ``sensor_to_body`` in
    input order; ``initial_error`` and ``final_error`` are the weighted
    objectives recomputed from the initial and final poses, so every field
    agrees with the pose it was derived from.
    """

    sensor_to_body: Pose3
    converged: bool
    iterations: int
    initial_error: float
    final_error: float
    residuals: tuple[tuple[float, float, float, float, float, float], ...]


# -- scalar / record validation ------------------------------------------------


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
        raise ValueError(f"{name} must be greater than zero, got {result!r}")
    return result


def _weight(value: object, name: str) -> float:
    """Validate one observation weight: a positive, finite, non-boolean real."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {result!r}")
    if result <= 0.0:
        raise ValueError(f"{name} must be greater than zero, got {result!r}")
    return result


def _pair(value: object, name: str, length: int) -> tuple:
    """Materialize one fixed-length record; type-error on bad shape."""
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be a sequence of {length} elements")
    try:
        items = tuple(value)  # type: ignore[call-overload]
    except TypeError:
        raise TypeError(f"{name} must be a sequence of {length} elements") from None
    if len(items) != length:
        raise TypeError(f"{name} must have exactly {length} elements, got {len(items)}")
    return items


def _unpack_observation(record: object, index: int) -> tuple[Pose3, Pose3, float]:
    """Validate one ``(body_motion, sensor_motion, weight)`` observation."""
    name = f"observations[{index}]"
    items = _pair(record, name, 3)
    body_motion = items[0]
    if not isinstance(body_motion, Pose3):
        raise TypeError(
            f"{name}[0] (body_motion) must be a Pose3, got "
            f"{type(body_motion).__name__}"
        )
    sensor_motion = items[1]
    if not isinstance(sensor_motion, Pose3):
        raise TypeError(
            f"{name}[1] (sensor_motion) must be a Pose3, got "
            f"{type(sensor_motion).__name__}"
        )
    return body_motion, sensor_motion, _weight(items[2], f"{name}[2] (weight)")


def _parse_observations(value: object) -> tuple[tuple[Pose3, Pose3, float], ...]:
    """Materialize and validate the observation sequence exactly once."""
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(
            "observations must be an iterable of "
            "(body_motion, sensor_motion, weight) triples"
        )
    try:
        raw_observations = list(value)  # type: ignore[call-overload]
    except TypeError:
        raise TypeError(
            "observations must be an iterable of "
            "(body_motion, sensor_motion, weight) triples"
        ) from None
    if len(raw_observations) < _MIN_OBSERVATIONS:
        raise ValueError(
            f"observations must contain at least {_MIN_OBSERVATIONS} records, "
            f"got {len(raw_observations)}"
        )
    return tuple(
        _unpack_observation(record, index)
        for index, record in enumerate(raw_observations)
    )


def _check_observability(observations: tuple[tuple[Pose3, Pose3, float], ...]) -> None:
    """Require the body-motion rotation axes to span at least two directions.

    The stacked residual Jacobians have full column rank six exactly when
    two body motions rotate about non-parallel axes (see the module
    docstring); anything less leaves at least one extrinsic degree of
    freedom unobservable no matter what the translations are.
    """
    reference_axis: tuple[float, float, float] | None = None
    for body_motion, _, _ in observations:
        xi = body_motion.log()
        angle = math.hypot(xi[0], xi[1], xi[2])
        if angle <= _AXIS_EPSILON:
            continue
        axis = (xi[0] / angle, xi[1] / angle, xi[2] / angle)
        if reference_axis is None:
            reference_axis = axis
            continue
        cross = (
            axis[1] * reference_axis[2] - axis[2] * reference_axis[1],
            axis[2] * reference_axis[0] - axis[0] * reference_axis[2],
            axis[0] * reference_axis[1] - axis[1] * reference_axis[0],
        )
        if math.hypot(*cross) > _AXIS_EPSILON:
            return
    raise ValueError(
        "observations cannot uniquely constrain the 6-DOF extrinsic: the "
        "body-motion rotation axes must span at least two distinct directions"
    )


# -- small dense linear algebra (standard library only) ------------------------


def _mat_mul(a, b):
    """Product of two square matrices stored as tuples of tuples."""
    size = len(a)
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(size)) for j in range(size))
        for i in range(size)
    )


def _mat_sub(a, b):
    """Difference of two same-shape matrices stored as tuples of tuples."""
    return tuple(
        tuple(a[i][j] - b[i][j] for j in range(len(a[i]))) for i in range(len(a))
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


# -- SE(3) adjoint and left Jacobian --------------------------------------------


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


# -- residuals, Jacobian, objective ----------------------------------------------


def _residual(body_motion: Pose3, sensor_motion: Pose3, extrinsic: Pose3):
    """``log( (A X)^{-1} (X B) )`` for one observation."""
    return (
        body_motion.compose(extrinsic)
        .inverse()
        .compose(extrinsic.compose(sensor_motion))
        .log()
    )


def _residuals(
    extrinsic: Pose3, observations: tuple[tuple[Pose3, Pose3, float], ...]
):
    """One 6D residual tangent vector per observation, in input order."""
    return tuple(
        _residual(body_motion, sensor_motion, extrinsic)
        for body_motion, sensor_motion, _ in observations
    )


def _jacobian(
    body_motion: Pose3, extrinsic: Pose3, residual
):
    """Exact 6x6 Jacobian ``d e / d d`` for ``X <- exp(d) X``.

    ``J = J_l(e)^{-1} ( Ad_{(A X)^{-1}} - Ad_{X^{-1}} )``; see the module
    docstring. The adjoint difference vanishes for identity body motions,
    which contribute no constraint.
    """
    adjoint_difference = _mat_sub(
        _adjoint(body_motion.compose(extrinsic).inverse()),
        _adjoint(extrinsic.inverse()),
    )
    return _mat_mul(_inverse_left_jacobian(residual), adjoint_difference)


def _total_error(
    extrinsic: Pose3, observations: tuple[tuple[Pose3, Pose3, float], ...]
) -> float:
    """Weighted objective ``sum(w * e . e)``; must stay finite."""
    total = 0.0
    for body_motion, sensor_motion, weight in observations:
        residual = _residual(body_motion, sensor_motion, extrinsic)
        total += weight * sum(component * component for component in residual)
    if not math.isfinite(total):
        raise ValueError("calibration objective is not finite")
    return total


# -- public entry point ----------------------------------------------------------


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
        Iterable of ``(body_motion, sensor_motion, weight)`` triples, at
        least three. Both motions are :class:`~mappilot.geometry.Pose3`
        relative poses over the same interval, each in its own reference
        frame; ``weight`` is a positive finite real. The iterable is
        consumed exactly once and caller objects are never mutated.
    initial_pose:
        Initial guess for the extrinsic :class:`~mappilot.geometry.Pose3`;
        the identity pose when omitted or ``None``.
    max_iterations:
        Maximum number of Gauss-Newton updates; positive integer,
        default 50.
    tolerance:
        Finite positive convergence threshold on the objective, default
        1e-6.

    Returns
    -------
    ExtrinsicCalibrationResult
        If the initial objective is already at most ``tolerance``, the
        result reports ``converged=True`` with ``iterations=0`` and the
        initial pose unchanged. Otherwise the loop stops as soon as the
        absolute objective change between two consecutive updates is at
        most ``tolerance`` (``converged=True``), or after
        ``max_iterations`` updates with ``converged=False``; exhausting
        the iteration budget never raises. ``residuals`` are the 6D
        residual tangent vectors at the final pose in input order, and
        ``initial_error``/``final_error`` are recomputed from the
        corresponding poses.

    Raises
    ------
    TypeError
        ``observations`` is not iterable, a record is not a three-element
        ``(body_motion, sensor_motion, weight)`` sequence, a motion or
        ``initial_pose`` is not a :class:`~mappilot.geometry.Pose3`, or a
        weight or control parameter has the wrong type (including
        booleans).
    ValueError
        Fewer than three observations; a weight that is non-positive or
        non-finite; ``max_iterations`` below one; a non-positive or
        non-finite ``tolerance``; a non-finite intermediate result; or a
        motion set that cannot uniquely constrain the six-degree-of-
        freedom extrinsic (all motions identity, all rotations zero or
        about a single axis).
    """
    iteration_limit = _positive_int(max_iterations, "max_iterations")
    convergence_tolerance = _positive_float(tolerance, "tolerance")
    if initial_pose is None:
        start = Pose3.identity()
    elif not isinstance(initial_pose, Pose3):
        raise TypeError(
            f"initial_pose must be a Pose3, got {type(initial_pose).__name__}"
        )
    else:
        start = initial_pose
    records = _parse_observations(observations)
    _check_observability(records)

    extrinsic = start
    initial_error = _total_error(extrinsic, records)
    if initial_error <= convergence_tolerance:
        return ExtrinsicCalibrationResult(
            extrinsic,
            True,
            0,
            initial_error,
            _total_error(extrinsic, records),
            _residuals(extrinsic, records),
        )

    converged = False
    iterations = 0
    previous_error = initial_error

    for _ in range(iteration_limit):
        hessian = [[0.0] * _DOF for _ in range(_DOF)]
        gradient = [0.0] * _DOF

        for body_motion, sensor_motion, weight in records:
            residual = _residual(body_motion, sensor_motion, extrinsic)
            jacobian = _jacobian(body_motion, extrinsic, residual)
            for row in range(_DOF):
                gradient[row] += weight * sum(
                    jacobian[k][row] * residual[k] for k in range(_DOF)
                )
            for row in range(_DOF):
                target = hessian[row]
                for col in range(_DOF):
                    target[col] += weight * sum(
                        jacobian[k][row] * jacobian[k][col] for k in range(_DOF)
                    )

        # Gauss-Newton step: H d = -J^T W e; apply X <- exp(d) X.
        step = _solve_linear(hessian, [-value for value in gradient])
        extrinsic = Pose3.exp(step).compose(extrinsic)
        iterations += 1

        current_error = _total_error(extrinsic, records)
        if abs(previous_error - current_error) <= convergence_tolerance:
            converged = True
            break
        previous_error = current_error

    return ExtrinsicCalibrationResult(
        extrinsic,
        converged,
        iterations,
        initial_error,
        _total_error(extrinsic, records),
        _residuals(extrinsic, records),
    )
