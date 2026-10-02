"""Rigid-body pose representation for MapPilot.

:class:`Pose3` is an SE(3) transform in a right-handed coordinate frame.
It maps coordinates expressed in a local frame into the parent frame:

    p_parent = R @ p_local + t

Rotations are stored as unit quaternions in ``(w, x, y, z)`` order and
translations as ``(x, y, z)``.  The module has no third-party dependencies
and never requires the HTTP service to be running.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from typing import Any

__all__ = ["Pose3"]

_TOL = 1e-9
_SMALL = 1e-6
_IDENTITY_TRANSLATION = (0.0, 0.0, 0.0)
_IDENTITY_QUATERNION = (1.0, 0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def _as_real(value: Any, where: str) -> float:
    """Return ``value`` as a finite float, mirroring the spec's error split."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(
            f"{where} must be a real number, got {type(value).__name__}"
        )
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{where} must be finite, got {result!r}")
    return result


def _as_sequence(value: Any, where: str) -> tuple[Any, ...]:
    if (
        isinstance(value, (str, bytes, bytearray))
        or isinstance(value, Mapping)
        or not hasattr(value, "__iter__")
    ):
        raise TypeError(f"{where} must be a sequence of real numbers")
    return tuple(value)


def _as_vec(value: Any, where: str, size: int) -> tuple[float, ...]:
    items = _as_sequence(value, where)
    if len(items) != size:
        raise TypeError(
            f"{where} must have length {size}, got {len(items)}"
        )
    return tuple(
        _as_real(item, f"{where}[{index}]") for index, item in enumerate(items)
    )


def _as_matrix4(value: Any, where: str) -> tuple[tuple[float, float, float, float], ...]:
    rows = _as_sequence(value, where)
    if len(rows) != 4:
        raise TypeError(f"{where} must have 4 rows, got {len(rows)}")
    matrix: list[tuple[float, float, float, float]] = []
    for row_index, row in enumerate(rows):
        row_where = f"{where}[{row_index}]"
        if (
            isinstance(row, (str, bytes, bytearray))
            or isinstance(row, Mapping)
            or not hasattr(row, "__iter__")
        ):
            raise TypeError(f"{row_where} must be a nested row of 4 real numbers")
        elements = tuple(row)
        if len(elements) != 4:
            raise TypeError(
                f"{row_where} must have length 4, got {len(elements)}"
            )
        matrix.append(
            tuple(
                _as_real(element, f"{row_where}[{column_index}]")
                for column_index, element in enumerate(elements)
            )
        )
    return tuple(matrix)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Quaternion / vector helpers (q is (w, x, y, z), unit length internally)
# ---------------------------------------------------------------------------


def _cross(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _rotate(
    q: tuple[float, float, float, float],
    v: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Rotate vector ``v`` by unit quaternion ``q`` (v' = q * v * q^{-1})."""
    w, x, y, z = q
    qv = (x, y, z)
    half = (
        2.0 * (y * v[2] - z * v[1]),
        2.0 * (z * v[0] - x * v[2]),
        2.0 * (x * v[1] - y * v[0]),
    )
    crossed = _cross(qv, half)
    return (
        v[0] + w * half[0] + crossed[0],
        v[1] + w * half[1] + crossed[1],
        v[2] + w * half[2] + crossed[2],
    )


def _quaternion_multiply(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Hamilton product; composition applies ``b`` then ``a`` on points."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _unit_canonical(
    q: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Normalize, then pick the representative with non-negative ``w``.

    When ``w`` is zero the first non-zero vector component is made positive,
    so q and -q always collapse to one stable quaternion.
    """
    norm = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    if norm == 0.0:
        raise ValueError("quaternion must have non-zero length")
    w, x, y, z = (component / norm for component in q)  # type: ignore[assignment]
    flip = w < 0.0
    if not flip and w == 0.0:
        for component in (x, y, z):
            if component != 0.0:
                flip = component < 0.0
                break
    if flip:
        w, x, y, z = -w, -x, -y, -z
    return w, x, y, z


def _rotation_matrix(
    q: tuple[float, float, float, float],
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    w, x, y, z = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return (
        (1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)),
        (2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)),
        (2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)),
    )


def _quaternion_from_rotation(
    r: tuple[tuple[float, float, float], ...],
) -> tuple[float, float, float, float]:
    """Shepperd's method: numerically stable for any rotation angle."""
    r00, r01, r02 = r[0]
    r10, r11, r12 = r[1]
    r20, r21, r22 = r[2]
    trace = r00 + r11 + r22
    if trace > 0.0:
        scale = 2.0 * math.sqrt(trace + 1.0)
        q = (
            0.25 * scale,
            (r21 - r12) / scale,
            (r02 - r20) / scale,
            (r10 - r01) / scale,
        )
    elif r00 >= r11 and r00 >= r22:
        scale = 2.0 * math.sqrt(1.0 + r00 - r11 - r22)
        q = (
            (r21 - r12) / scale,
            0.25 * scale,
            (r01 + r10) / scale,
            (r02 + r20) / scale,
        )
    elif r11 > r22:
        scale = 2.0 * math.sqrt(1.0 + r11 - r00 - r22)
        q = (
            (r02 - r20) / scale,
            (r01 + r10) / scale,
            0.25 * scale,
            (r12 + r21) / scale,
        )
    else:
        scale = 2.0 * math.sqrt(1.0 + r22 - r00 - r11)
        q = (
            (r10 - r01) / scale,
            (r02 + r20) / scale,
            (r12 + r21) / scale,
            0.25 * scale,
        )
    return _unit_canonical(q)


def _decompose_matrix(
    matrix: tuple[tuple[float, float, float, float], ...],
) -> tuple[
    tuple[float, float, float], tuple[float, float, float, float]
]:
    rotation = tuple(row[:3] for row in matrix[:3])
    # Rows must be orthonormal ...
    for i in range(3):
        for j in range(i, 3):
            dot_product = sum(rotation[i][k] * rotation[j][k] for k in range(3))
            expected = 1.0 if i == j else 0.0
            if abs(dot_product - expected) > _TOL:
                raise ValueError(
                    "rotation block must be orthogonal within 1e-9 absolute error"
                )
    # ... and right-handed (reject reflections instead of "fixing" them).
    determinant = (
        rotation[0][0] * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
        - rotation[0][1] * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
        + rotation[0][2] * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
    )
    if determinant <= 0.0:
        raise ValueError("rotation block must have positive determinant")
    quaternion = _quaternion_from_rotation(rotation)
    translation = (matrix[0][3], matrix[1][3], matrix[2][3])
    return translation, quaternion


# ---------------------------------------------------------------------------
# Pose3
# ---------------------------------------------------------------------------


class Pose3:
    """Immutable SE(3) pose mapping local coordinates into a parent frame."""

    __slots__ = ("_t", "_q")

    def __init__(
        self,
        translation: Any = None,
        quaternion: Any = None,
        *,
        matrix: Any = None,
    ) -> None:
        if matrix is not None:
            if translation is not None or quaternion is not None:
                raise TypeError(
                    "use either translation/quaternion or matrix, not both"
                )
            parsed = _as_matrix4(matrix, "matrix")
            bottom = parsed[3]
            expected_bottom = (0.0, 0.0, 0.0, 1.0)
            if any(
                abs(bottom[i] - expected_bottom[i]) > _TOL for i in range(4)
            ):
                raise ValueError(
                    "homogeneous matrix last row must be (0, 0, 0, 1) within 1e-9"
                )
            self._t, self._q = _decompose_matrix(parsed)
            return
        self._t = (
            _IDENTITY_TRANSLATION
            if translation is None
            else _as_vec(translation, "translation", 3)
        )
        raw_quaternion = (
            _IDENTITY_QUATERNION if quaternion is None else _as_vec(quaternion, "quaternion", 4)
        )
        self._q = _unit_canonical(raw_quaternion)

    @classmethod
    def _make(
        cls,
        translation: tuple[float, float, float],
        quaternion: tuple[float, float, float, float],
    ) -> "Pose3":
        """Construct from already validated data (quaternion still normalized)."""
        pose = super().__new__(cls)
        pose._t = (float(translation[0]), float(translation[1]), float(translation[2]))
        pose._q = _unit_canonical(quaternion)
        return pose

    @classmethod
    def from_tq(cls, translation: Any, quaternion: Any) -> "Pose3":
        """Create from an ``(x, y, z)`` translation and ``(w, x, y, z)`` quaternion."""
        return cls(translation, quaternion)

    @classmethod
    def from_translation_quaternion(cls, translation: Any, quaternion: Any) -> "Pose3":
        return cls(translation, quaternion)

    @classmethod
    def from_matrix(cls, matrix: Any) -> "Pose3":
        """Create from a validated 4x4 homogeneous matrix."""
        return cls(matrix=matrix)

    @classmethod
    def identity(cls) -> "Pose3":
        return cls._make(_IDENTITY_TRANSLATION, _IDENTITY_QUATERNION)

    @classmethod
    def exp(cls, tangent: Any) -> "Pose3":
        """SE(3) exponential.

        ``tangent`` is ``(wx, wy, wz, vx, vy, vz)`` under the left-perturbation
        convention: the rotation vector comes first and the translation is
        mapped through the left Jacobian ``J_l(w) @ v``.
        """
        values = _as_vec(tangent, "tangent", 6)
        omega = values[:3]
        velocity = values[3:]
        theta = math.sqrt(omega[0] ** 2 + omega[1] ** 2 + omega[2] ** 2)
        if theta == 0.0:
            return cls._make(velocity, _IDENTITY_QUATERNION)
        if theta < _SMALL:
            theta2 = theta * theta
            theta4 = theta2 * theta2
            beta = 0.5 - theta2 / 24.0 + theta4 / 720.0
            delta = 1.0 / 6.0 - theta2 / 120.0 + theta4 / 5040.0
            qw = 1.0 - theta2 / 8.0 + theta4 / 384.0
            scale = 0.5 - theta2 / 48.0 + theta4 / 3840.0
        else:
            beta = (1.0 - math.cos(theta)) / (theta * theta)
            delta = (theta - math.sin(theta)) / (theta ** 3)
            qw = math.cos(0.5 * theta)
            scale = math.sin(0.5 * theta) / theta
        quaternion = (
            qw,
            scale * omega[0],
            scale * omega[1],
            scale * omega[2],
        )
        cross = _cross(omega, velocity)
        cross2 = _cross(omega, cross)  # [w]_x^2 v
        translation = (
            velocity[0] + beta * cross[0] + delta * cross2[0],
            velocity[1] + beta * cross[1] + delta * cross2[1],
            velocity[2] + beta * cross[2] + delta * cross2[2],
        )
        return cls._make(translation, quaternion)

    # -- observers ---------------------------------------------------------

    @property
    def translation(self) -> tuple[float, float, float]:
        return self._t

    @property
    def quaternion(self) -> tuple[float, float, float, float]:
        """Unit, sign-canonical quaternion ``(w, x, y, z)``."""
        return self._q

    @property
    def rotation(self) -> tuple[float, float, float, float]:
        return self._q

    def to_matrix(self) -> tuple[tuple[float, float, float, float], ...]:
        """Fresh 4x4 homogeneous matrix as nested immutable tuples."""
        r = _rotation_matrix(self._q)
        tx, ty, tz = self._t
        return (
            (r[0][0], r[0][1], r[0][2], tx),
            (r[1][0], r[1][1], r[1][2], ty),
            (r[2][0], r[2][1], r[2][2], tz),
            (0.0, 0.0, 0.0, 1.0),
        )

    def as_matrix(self) -> tuple[tuple[float, float, float, float], ...]:
        return self.to_matrix()

    # -- SE(3) algebra -----------------------------------------------------

    def compose(self, other: "Pose3") -> "Pose3":
        """Return ``A @ B`` as transforms: ``A.compose(B)`` applies B first."""
        if not isinstance(other, Pose3):
            raise TypeError(
                f"can only compose Pose3 with Pose3, got {type(other).__name__}"
            )
        rotated = _rotate(self._q, other._t)
        translation = (
            self._t[0] + rotated[0],
            self._t[1] + rotated[1],
            self._t[2] + rotated[2],
        )
        return self._make(translation, _quaternion_multiply(self._q, other._q))

    def inverse(self) -> "Pose3":
        w, x, y, z = self._q
        conjugate = (w, -x, -y, -z)
        negated = (-self._t[0], -self._t[1], -self._t[2])
        translation = _rotate(conjugate, negated)
        return self._make(translation, conjugate)

    def transform(self, point: Any) -> tuple[float, float, float]:
        """Transform one local-frame 3D point into the parent frame."""
        local = _as_vec(point, "point", 3)
        rotated = _rotate(self._q, local)
        return (
            self._t[0] + rotated[0],
            self._t[1] + rotated[1],
            self._t[2] + rotated[2],
        )

    def transform_point(self, point: Any) -> tuple[float, float, float]:
        return self.transform(point)

    def transform_points(self, points: Any) -> tuple[tuple[float, float, float], ...]:
        """Transform a sequence of 3D points; order/count preserved.

        An empty sequence returns an empty tuple.
        """
        items = _as_sequence(points, "points")
        return tuple(self.transform(point) for point in items)

    def log(self) -> tuple[float, float, float, float, float, float]:
        """Inverse of :meth:`exp`; rotation vector first, then translation.

        The angle is read straight from the canonical unit quaternion, so it
        stays in ``[0, pi]`` and, near ``pi``, the axis inherits the quaternion
        sign convention (w positive; if w is zero the first non-zero vector
        component is positive) instead of being recovered from the
        ill-conditioned skew part of the rotation matrix.
        """
        w, qx, qy, qz = self._q
        sin_half = math.sqrt(qx * qx + qy * qy + qz * qz)
        theta = 2.0 * math.atan2(sin_half, w)
        if theta == 0.0:
            omega: tuple[float, float, float] = (0.0, 0.0, 0.0)
            coefficient = 1.0 / 12.0
        elif theta < _SMALL:
            # q_vec/|q_vec| is ill-conditioned near zero; use the skew part of
            # R with the theta/(2 sin theta) series instead.
            r = _rotation_matrix(self._q)
            factor = 0.5 + theta * theta / 12.0 + 7.0 * theta ** 4 / 720.0
            omega = (
                factor * (r[2][1] - r[1][2]),
                factor * (r[0][2] - r[2][0]),
                factor * (r[1][0] - r[0][1]),
            )
            coefficient = 1.0 / 12.0 + theta * theta / 720.0
        else:
            inverse_half = 1.0 / sin_half
            omega = (
                theta * qx * inverse_half,
                theta * qy * inverse_half,
                theta * qz * inverse_half,
            )
            # (theta/2) cot(theta/2) = (theta/2) * w / sin(theta/2)
            half_cot = 0.5 * theta * w * inverse_half
            coefficient = (1.0 - half_cot) / (theta * theta)
        if theta == 0.0:
            velocity = self._t
        else:
            # v = J_l(w)^{-1} t = t - 1/2 w x t + c2 [w]_x^2 t
            cross = _cross(omega, self._t)
            cross2 = _cross(omega, cross)
            velocity = (
                self._t[0] - 0.5 * cross[0] + coefficient * cross2[0],
                self._t[1] - 0.5 * cross[1] + coefficient * cross2[1],
                self._t[2] - 0.5 * cross[2] + coefficient * cross2[2],
            )
        return (
            omega[0], omega[1], omega[2],
            velocity[0], velocity[1], velocity[2],
        )

    # -- dunder conveniences ----------------------------------------------

    def __matmul__(self, other: "Pose3") -> "Pose3":
        if not isinstance(other, Pose3):
            return NotImplemented
        return self.compose(other)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Pose3):
            return NotImplemented
        return self._t == other._t and self._q == other._q

    def __hash__(self) -> int:
        return hash((self._t, self._q))

    def __repr__(self) -> str:
        return f"Pose3(translation={self._t!r}, quaternion={self._q!r})"
