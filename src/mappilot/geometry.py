"""SE(3) rigid-body poses shared by odometry, scan matching, and graph optimization.

This module is pure Python and dependency-free: importing it never requires
the HTTP service. Conventions:

* Right-handed coordinates. A pose maps local coordinates to parent
  coordinates, i.e. ``p_parent = R * p_local + t``.
* Translation is ordered ``(x, y, z)``; quaternions are ordered
  ``(w, x, y, z)``.
* ``q`` and ``-q`` describe the same rotation. Every public quaternion
  output is canonicalized: ``w`` is positive; when ``w`` is zero, the first
  non-zero vector component is positive, so one rotation always yields one
  stable quaternion.
* Six-dimensional tangent vectors are ordered ``(wx, wy, wz, vx, vy, vz)``:
  rotation vector first, then translation increment. :meth:`Pose3.exp` and
  :meth:`Pose3.log` follow the left-perturbation convention and are mutual
  inverses; the rotation angle returned by :meth:`Pose3.log` lies in
  ``[0, pi]`` and stays continuous near zero angle.

Validation policy: non-real scalars (including booleans), wrong dimensions,
and wrong nested shapes raise :class:`TypeError`; NaN or infinite values,
zero-length quaternions, homogeneous matrices whose last row is not
``(0, 0, 0, 1)``, and rotation blocks that are not orthogonal with
determinant +1 raise :class:`ValueError`. Matrix checks use an absolute
tolerance of 1e-9. Reflections and clearly non-orthogonal blocks are
rejected, never silently repaired.
"""

from __future__ import annotations

import math
import numbers
from typing import Iterable

__all__ = ["Pose3"]

# Absolute tolerance for homogeneous-matrix validation.
_MATRIX_TOL = 1e-9

# Below this rotation angle (radians), series expansions replace the closed
# forms of the SE(3) exp/log coefficients to avoid catastrophic cancellation.
_SMALL_ANGLE = 0.01

# A canonical quaternion with |w| below this threshold represents a rotation
# whose angle is pi to within floating-point noise (cos(pi/2) evaluates to
# ~6e-17). log() then picks the rotation axis by the canonical sign rule
# instead of letting rounding noise flip it.
_PI_EPSILON = 1e-15

_IDENTITY_QUATERNION = (1.0, 0.0, 0.0, 0.0)
_ZERO_TRANSLATION = (0.0, 0.0, 0.0)


def _scalar(value: object, name: str) -> float:
    """Coerce one real, finite, non-boolean scalar to float."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {result!r}")
    return result


def _vector(value: object, length: int, name: str) -> tuple[float, ...]:
    """Coerce a sequence of exactly ``length`` real scalars to a tuple."""
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be a sequence of {length} real numbers")
    try:
        items = tuple(value)  # type: ignore[call-overload]
    except TypeError:
        raise TypeError(f"{name} must be a sequence of {length} real numbers") from None
    if len(items) != length:
        raise TypeError(f"{name} must have {length} elements, got {len(items)}")
    return tuple(_scalar(item, f"{name}[{index}]") for index, item in enumerate(items))


def _canonical_quaternion(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Pick one stable representative out of ``q`` and ``-q``."""
    w, x, y, z = q
    if w < 0.0:
        return (-w, -x, -y, -z)
    if w == 0.0:
        for component in (x, y, z):
            if component != 0.0:
                if component < 0.0:
                    return (-w, -x, -y, -z)
                break
    return q


def _normalize_quaternion(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    norm = math.hypot(*q)
    if norm == 0.0:
        raise ValueError("quaternion must have non-zero length")
    return _canonical_quaternion((q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm))


def _quaternion_multiply(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _quaternion_to_matrix(
    q: tuple[float, float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    w, x, y, z = q
    return (
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)),
        (2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)),
        (2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)),
    )


def _matrix_to_quaternion(
    r: tuple[tuple[float, float, float], ...],
) -> tuple[float, float, float, float]:
    (r00, r01, r02), (r10, r11, r12), (r20, r21, r22) = r
    trace = r00 + r11 + r22
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return _normalize_quaternion((0.25 * s, (r21 - r12) / s, (r02 - r20) / s, (r10 - r01) / s))
    if r00 > r11 and r00 > r22:
        s = math.sqrt(1.0 + r00 - r11 - r22) * 2.0
        return _normalize_quaternion(((r21 - r12) / s, 0.25 * s, (r01 + r10) / s, (r02 + r20) / s))
    if r11 > r22:
        s = math.sqrt(1.0 + r11 - r00 - r22) * 2.0
        return _normalize_quaternion(((r02 - r20) / s, (r01 + r10) / s, 0.25 * s, (r12 + r21) / s))
    s = math.sqrt(1.0 + r22 - r00 - r11) * 2.0
    return _normalize_quaternion(((r10 - r01) / s, (r02 + r20) / s, (r12 + r21) / s, 0.25 * s))


def _cross(
    a: tuple[float, float, float], b: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _validate_homogeneous_matrix(
    value: object,
) -> tuple[tuple[tuple[float, float, float], ...], tuple[float, float, float]]:
    """Validate a 4x4 homogeneous matrix; return (rotation rows, translation)."""
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError("matrix must be a 4x4 nested sequence of real numbers")
    try:
        rows = tuple(value)  # type: ignore[call-overload]
    except TypeError:
        raise TypeError("matrix must be a 4x4 nested sequence of real numbers") from None
    if len(rows) != 4:
        raise TypeError(f"matrix must have 4 rows, got {len(rows)}")
    grid = tuple(_vector(row, 4, f"matrix[{index}]") for index, row in enumerate(rows))

    last = grid[3]
    for index, expected in enumerate((0.0, 0.0, 0.0, 1.0)):
        if abs(last[index] - expected) > _MATRIX_TOL:
            raise ValueError("matrix last row must be (0, 0, 0, 1)")

    rotation = tuple(row[:3] for row in grid[:3])
    for i in range(3):
        for j in range(3):
            expected = 1.0 if i == j else 0.0
            dot = sum(rotation[k][i] * rotation[k][j] for k in range(3))
            if abs(dot - expected) > _MATRIX_TOL:
                raise ValueError("matrix rotation block is not orthogonal")
    (r00, r01, r02), (r10, r11, r12), (r20, r21, r22) = rotation
    det = (
        r00 * (r11 * r22 - r12 * r21)
        - r01 * (r10 * r22 - r12 * r20)
        + r02 * (r10 * r21 - r11 * r20)
    )
    if abs(det - 1.0) > _MATRIX_TOL:
        raise ValueError("matrix rotation block must have determinant +1")

    translation = (grid[0][3], grid[1][3], grid[2][3])
    return rotation, translation


class Pose3:
    """Rigid-body pose in SE(3): transforms local coordinates to parent coordinates.

    Create from a translation and a quaternion (both copied on construction,
    so later mutation of the inputs cannot affect the pose), or from a 4x4
    homogeneous matrix via :meth:`from_matrix`. All returned sequences are
    immutable tuples.
    """

    __slots__ = ("_translation", "_quaternion")

    def __init__(
        self,
        translation: Iterable[float] = _ZERO_TRANSLATION,
        quaternion: Iterable[float] = _IDENTITY_QUATERNION,
    ) -> None:
        t = _vector(translation, 3, "translation")
        q = _normalize_quaternion(_vector(quaternion, 4, "quaternion"))
        self._translation = t
        self._quaternion = q

    # -- constructors ------------------------------------------------------

    @classmethod
    def identity(cls) -> Pose3:
        """Return the identity pose."""
        return cls()

    @classmethod
    def from_matrix(cls, matrix: Iterable[Iterable[float]]) -> Pose3:
        """Create a pose from a 4x4 homogeneous matrix.

        The last row must be ``(0, 0, 0, 1)`` and the rotation block must be
        orthogonal with determinant +1 (absolute tolerance 1e-9).
        """
        rotation, translation = _validate_homogeneous_matrix(matrix)
        return cls(translation, _matrix_to_quaternion(rotation))

    @classmethod
    def exp(cls, xi: Iterable[float]) -> Pose3:
        """Map a 6D tangent vector ``(wx, wy, wz, vx, vy, vz)`` to a pose.

        Left-perturbation convention: the rotation vector builds the
        rotation via Rodrigues' formula and the translation increment is
        mapped through the left Jacobian of SO(3). Inverse of :meth:`log`.
        """
        wx, wy, wz, vx, vy, vz = _vector(xi, 6, "xi")
        omega = (wx, wy, wz)
        theta = math.hypot(wx, wy, wz)
        if theta == 0.0:
            return cls((vx, vy, vz), _IDENTITY_QUATERNION)

        half = 0.5 * theta
        k = math.sin(half) / theta
        quaternion = (math.cos(half), k * wx, k * wy, k * wz)

        # V = I + A*[w]x + B*[w]x^2 maps the translation increment.
        if theta < _SMALL_ANGLE:
            t2 = theta * theta
            a = 0.5 - t2 / 24.0 + t2 * t2 / 720.0
            b = 1.0 / 6.0 - t2 / 120.0 + t2 * t2 / 5040.0
        else:
            a = (1.0 - math.cos(theta)) / (theta * theta)
            b = (theta - math.sin(theta)) / (theta * theta * theta)
        v = (vx, vy, vz)
        w_cross_v = _cross(omega, v)
        w_cross_w_cross_v = _cross(omega, w_cross_v)
        translation = (
            vx + a * w_cross_v[0] + b * w_cross_w_cross_v[0],
            vy + a * w_cross_v[1] + b * w_cross_w_cross_v[1],
            vz + a * w_cross_v[2] + b * w_cross_w_cross_v[2],
        )
        return cls(translation, quaternion)

    # -- accessors ---------------------------------------------------------

    @property
    def translation(self) -> tuple[float, float, float]:
        """Translation ``(x, y, z)`` as an immutable tuple."""
        return self._translation

    @property
    def quaternion(self) -> tuple[float, float, float, float]:
        """Unit quaternion ``(w, x, y, z)``, canonicalized to ``w >= 0``."""
        return self._quaternion

    @property
    def rotation(self) -> tuple[tuple[float, float, float], ...]:
        """3x3 rotation matrix as nested immutable tuples."""
        return _quaternion_to_matrix(self._quaternion)

    @property
    def matrix(self) -> tuple[tuple[float, float, float, float], ...]:
        """4x4 homogeneous matrix as nested immutable tuples."""
        r = _quaternion_to_matrix(self._quaternion)
        t = self._translation
        return (
            (r[0][0], r[0][1], r[0][2], t[0]),
            (r[1][0], r[1][1], r[1][2], t[1]),
            (r[2][0], r[2][1], r[2][2], t[2]),
            (0.0, 0.0, 0.0, 1.0),
        )

    # -- group operations --------------------------------------------------

    def compose(self, other: Pose3) -> Pose3:
        """Return ``self * other``: applies ``other`` first, then ``self``."""
        if not isinstance(other, Pose3):
            raise TypeError(f"other must be a Pose3, got {type(other).__name__}")
        r = _quaternion_to_matrix(self._quaternion)
        ot = other._translation
        st = self._translation
        translation = (
            r[0][0] * ot[0] + r[0][1] * ot[1] + r[0][2] * ot[2] + st[0],
            r[1][0] * ot[0] + r[1][1] * ot[1] + r[1][2] * ot[2] + st[1],
            r[2][0] * ot[0] + r[2][1] * ot[1] + r[2][2] * ot[2] + st[2],
        )
        quaternion = _normalize_quaternion(_quaternion_multiply(self._quaternion, other._quaternion))
        return Pose3(translation, quaternion)

    def inverse(self) -> Pose3:
        """Return the inverse pose; ``self.compose(self.inverse())`` is identity."""
        w, x, y, z = self._quaternion
        r = _quaternion_to_matrix((w, -x, -y, -z))  # transposed rotation
        t = self._translation
        translation = (
            -(r[0][0] * t[0] + r[0][1] * t[1] + r[0][2] * t[2]),
            -(r[1][0] * t[0] + r[1][1] * t[1] + r[1][2] * t[2]),
            -(r[2][0] * t[0] + r[2][1] * t[1] + r[2][2] * t[2]),
        )
        return Pose3(translation, (w, -x, -y, -z))

    def log(self) -> tuple[float, float, float, float, float, float]:
        """Map this pose to its 6D tangent vector ``(wx, wy, wz, vx, vy, vz)``.

        Inverse of :meth:`exp`. The rotation angle lies in ``[0, pi]``; the
        axis follows the quaternion canonicalization rule, so it never flips
        arbitrarily near angle pi.
        """
        w, x, y, z = self._quaternion
        s = math.hypot(x, y, z)
        if s == 0.0:
            omega = (0.0, 0.0, 0.0)
            theta = 0.0
        else:
            theta = 2.0 * math.atan2(s, w)
            if w < _PI_EPSILON:
                # Angle is pi within float noise: the axis sign is ambiguous,
                # so apply the canonical rule (first non-zero vector
                # component positive) instead of trusting the sign of w.
                for component in (x, y, z):
                    if component != 0.0:
                        if component < 0.0:
                            x, y, z = -x, -y, -z
                        break
            scale = theta / s
            omega = (scale * x, scale * y, scale * z)

        # J^-1 = I - 0.5*[w]x + c*[w]x^2 maps the translation back.
        if theta < _SMALL_ANGLE:
            t2 = theta * theta
            c = 1.0 / 12.0 + t2 / 720.0 + t2 * t2 / 30240.0
        else:
            c = 1.0 / (theta * theta) - math.cos(0.5 * theta) / (
                2.0 * theta * math.sin(0.5 * theta)
            )
        t = self._translation
        w_cross_t = _cross(omega, t)
        w_cross_w_cross_t = _cross(omega, w_cross_t)
        v = (
            t[0] - 0.5 * w_cross_t[0] + c * w_cross_w_cross_t[0],
            t[1] - 0.5 * w_cross_t[1] + c * w_cross_w_cross_t[1],
            t[2] - 0.5 * w_cross_t[2] + c * w_cross_w_cross_t[2],
        )
        return (omega[0], omega[1], omega[2], v[0], v[1], v[2])

    # -- point transforms --------------------------------------------------

    def transform_point(self, point: Iterable[float]) -> tuple[float, float, float]:
        """Transform one 3D point from local to parent coordinates."""
        px, py, pz = _vector(point, 3, "point")
        r = _quaternion_to_matrix(self._quaternion)
        t = self._translation
        return (
            r[0][0] * px + r[0][1] * py + r[0][2] * pz + t[0],
            r[1][0] * px + r[1][1] * py + r[1][2] * pz + t[1],
            r[2][0] * px + r[2][1] * py + r[2][2] * pz + t[2],
        )

    def transform_points(
        self, points: Iterable[Iterable[float]]
    ) -> tuple[tuple[float, float, float], ...]:
        """Transform a sequence of 3D points, preserving order and count.

        An empty sequence yields an empty tuple.
        """
        if isinstance(points, (str, bytes, bytearray)):
            raise TypeError("points must be a sequence of 3D points")
        return tuple(self.transform_point(point) for point in points)

    # -- dunder ------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Pose3):
            return NotImplemented
        return self._translation == other._translation and self._quaternion == other._quaternion

    def __hash__(self) -> int:
        return hash((self._translation, self._quaternion))

    def __repr__(self) -> str:
        return f"Pose3(translation={self._translation!r}, quaternion={self._quaternion!r})"
