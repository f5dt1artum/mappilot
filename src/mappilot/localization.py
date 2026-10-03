"""Deterministic correlative scan matching on a 2D occupancy grid.

The :func:`correlative_scan_match` entry point relocalizes a local 3D scan
against a :class:`~mappilot.mapping.OccupancyGrid2D` when a rough prior is
available (relocalization and lost recovery): it exhaustively scores a
regular grid of candidate poses around the initial guess and returns the
best one.

Candidate poses are built from the initial pose by adding ``dx``/``dy``
directly to the initial world translation (z is kept unchanged) and
left-composing a rotation of ``dyaw`` about the world z axis onto the
initial rotation. Each offset is an integer multiple of the matching step
whose absolute value does not exceed the matching window; zero is always
included, so the initial pose itself is always a candidate.

Every candidate transforms the scan with :class:`~mappilot.geometry.Pose3`
semantics and projects the transformed points onto the grid with the
``floor((p - origin) / resolution)`` cell rule; the z coordinate never
takes part in scoring. A point landing on an occupied cell contributes +1,
on a free cell -1; points on unknown cells or outside the map contribute
nothing and are not counted as known. Candidates with at least
``min_known_points`` known points are scored by the mean of those
contributions; all other candidates are discarded.

The winner is the highest-scoring candidate; exact score ties resolve to
more known points, then to the smaller ``dx**2 + dy**2 + dyaw**2``, then to
ascending ``dx``, ``dy``, ``dyaw``. The immutable result exposes ``pose``,
``matched``, ``score``, ``known_points`` and ``evaluated_candidates``.
``matched`` is True exactly when the best score reaches ``min_score``; a
best candidate below the threshold is still returned with its pose and
score. When no candidate reaches ``min_known_points`` the result carries
the original initial pose, ``matched=False``, ``score=None`` and
``known_points=0``. ``evaluated_candidates`` always counts every candidate
actually searched.

This module is pure Python and depends only on the standard library plus
:mod:`mappilot.geometry` and :mod:`mappilot.mapping`. Importing it never
starts the HTTP service.

Validation policy mirrors :mod:`mappilot.mapping`: a non-iterable scan,
wrong point dimensions, non-real (including boolean) coordinates or
parameters, a non-integer (including boolean) ``min_known_points``, and a
grid or initial pose of the wrong type raise :class:`TypeError`; an empty
scan, non-finite coordinates, negative or non-finite windows, non-positive
or non-finite steps, a ``min_known_points`` below one, and a ``min_score``
outside ``[-1, 1]`` raise :class:`ValueError`.
"""

from __future__ import annotations

import math
import numbers
from typing import Iterable, NamedTuple

from .geometry import Pose3
from .mapping import FREE, OCCUPIED, OccupancyGrid2D

__all__ = ["CorrelativeScanMatchResult", "correlative_scan_match"]


class CorrelativeScanMatchResult(NamedTuple):
    """Outcome of :func:`correlative_scan_match`.

    Immutable. ``pose`` is the best candidate (or the original initial pose
    when no candidate reached ``min_known_points``); ``matched`` is True
    exactly when ``score`` reaches ``min_score``; ``score`` is the mean
    cell contribution of the best candidate (``None`` when no candidate
    qualified); ``known_points`` is its known-point count (0 when none
    qualified); ``evaluated_candidates`` counts every candidate searched.
    """

    pose: Pose3
    matched: bool
    score: float | None
    known_points: int
    evaluated_candidates: int


# -- validation -------------------------------------------------------------


def _real_scalar(value: object, name: str) -> float:
    """Coerce one real, finite, non-boolean scalar to float."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {result!r}")
    return result


def _scan_points(value: object) -> tuple[tuple[float, float, float], ...]:
    """Validate a non-empty scan of 3D finite points, consumed exactly once."""
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError("scan_points must be an iterable of 3D points")
    try:
        raw_points = tuple(value)  # type: ignore[call-overload]
    except TypeError:
        raise TypeError("scan_points must be an iterable of 3D points") from None
    if len(raw_points) == 0:
        raise ValueError("scan_points must contain at least one point, got 0")

    points: list[tuple[float, float, float]] = []
    for index, raw_point in enumerate(raw_points):
        name = f"scan_points[{index}]"
        if isinstance(raw_point, (str, bytes, bytearray)):
            raise TypeError(f"{name} must be a sequence of 3 real numbers")
        try:
            raw_coords = tuple(raw_point)  # type: ignore[call-overload]
        except TypeError:
            raise TypeError(f"{name} must be a sequence of 3 real numbers") from None
        if len(raw_coords) != 3:
            raise TypeError(f"{name} must have 3 coordinates, got {len(raw_coords)}")
        point = tuple(
            _real_scalar(coord, f"{name}[{coord_index}]")
            for coord_index, coord in enumerate(raw_coords)
        )
        points.append(point)  # type: ignore[arg-type]
    return tuple(points)


def _window(value: object, name: str) -> float:
    result = _real_scalar(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative, got {result}")
    return result


def _step(value: object, name: str) -> float:
    result = _real_scalar(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be greater than zero, got {result}")
    return result


def _min_known_points(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(
            f"min_known_points must be an integer, got {type(value).__name__}"
        )
    result = int(value)
    if result < 1:
        raise ValueError(f"min_known_points must be at least 1, got {result}")
    return result


def _min_score(value: object) -> float:
    result = _real_scalar(value, "min_score")
    if not -1.0 <= result <= 1.0:
        raise ValueError(f"min_score must lie in [-1, 1], got {result}")
    return result


# -- candidate generation -----------------------------------------------------


def _offsets(window: float, step: float) -> tuple[float, ...]:
    """Integer step multiples with absolute value not exceeding ``window``.

    Zero is always included (``step`` is positive, ``window`` non-negative).
    """
    count = int(math.floor(window / step))
    return tuple(
        offset
        for offset in (k * step for k in range(-count, count + 1))
        if abs(offset) <= window
    )


def _yaw_quaternion(
    dyaw: float, q: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    """Left-compose a world-z rotation of ``dyaw`` onto quaternion ``q``."""
    half = 0.5 * dyaw
    c = math.cos(half)
    s = math.sin(half)
    w, x, y, z = q
    return (c * w - s * z, c * x - s * y, c * y + s * x, c * z + s * w)


# -- public entry point -------------------------------------------------------


def correlative_scan_match(
    grid: OccupancyGrid2D,
    scan_points: Iterable[Iterable[float]],
    initial_pose: Pose3,
    x_window: float = 0.0,
    x_step: float = 0.05,
    y_window: float = 0.0,
    y_step: float = 0.05,
    yaw_window: float = 0.0,
    yaw_step: float = 0.1,
    min_known_points: int = 1,
    min_score: float = 0.0,
) -> CorrelativeScanMatchResult:
    """Exhaustively match a local scan against a 2D occupancy grid.

    Parameters
    ----------
    grid:
        The :class:`~mappilot.mapping.OccupancyGrid2D` to match against.
    scan_points:
        Non-empty iterable of finite ``(x, y, z)`` points in the local
        sensor frame. Consumed exactly once and never modified.
    initial_pose:
        Rough prior :class:`~mappilot.geometry.Pose3`; every candidate adds
        ``dx``/``dy`` to its world translation (z unchanged) and
        left-composes a world-z rotation of ``dyaw`` onto its rotation.
    x_window, y_window, yaw_window:
        Non-negative finite search half-ranges around the initial pose
        (yaw in radians), default 0.0.
    x_step, y_step, yaw_step:
        Strictly positive finite offset steps; each offset is an integer
        multiple of the step with absolute value not exceeding the window,
        and zero is always included. Defaults 0.05, 0.05 and 0.1.
    min_known_points:
        Minimum known-point count for a candidate to be scored; a
        non-boolean integer of at least 1, default 1.
    min_score:
        Acceptance threshold in ``[-1, 1]``, default 0.0; the best
        candidate is ``matched`` exactly when its score reaches it.

    Returns
    -------
    CorrelativeScanMatchResult
        The highest-scoring candidate (ties: more known points, then
        smaller ``dx**2 + dy**2 + dyaw**2``, then ascending ``dx``, ``dy``,
        ``dyaw``). A best candidate below ``min_score`` is still returned
        with ``matched=False``. When no candidate reaches
        ``min_known_points``, the result carries the original initial pose,
        ``matched=False``, ``score=None`` and ``known_points=0``.
        ``evaluated_candidates`` always counts every candidate searched.
    """
    if not isinstance(grid, OccupancyGrid2D):
        raise TypeError(
            f"grid must be an OccupancyGrid2D, got {type(grid).__name__}"
        )
    if not isinstance(initial_pose, Pose3):
        raise TypeError(
            f"initial_pose must be a Pose3, got {type(initial_pose).__name__}"
        )
    scan = _scan_points(scan_points)
    x_win = _window(x_window, "x_window")
    x_st = _step(x_step, "x_step")
    y_win = _window(y_window, "y_window")
    y_st = _step(y_step, "y_step")
    yaw_win = _window(yaw_window, "yaw_window")
    yaw_st = _step(yaw_step, "yaw_step")
    min_known = _min_known_points(min_known_points)
    score_threshold = _min_score(min_score)

    x_offsets = _offsets(x_win, x_st)
    y_offsets = _offsets(y_win, y_st)
    yaw_offsets = _offsets(yaw_win, yaw_st)

    origin_x, origin_y = grid.origin
    resolution = grid.resolution
    width = grid.width
    height = grid.height
    data = grid.data

    initial_translation = initial_pose.translation
    initial_quaternion = initial_pose.quaternion

    evaluated = 0
    best_key: tuple[float, float, float, float, float, float] | None = None
    best_offsets: tuple[float, float, float] | None = None
    best_score = 0.0
    best_known = 0

    for dyaw in yaw_offsets:
        # The candidate rotation depends only on dyaw; building the pose
        # once per dyaw keeps scoring exactly consistent with the Pose3
        # returned for the winning candidate.
        rotation = Pose3(
            (0.0, 0.0, 0.0), _yaw_quaternion(dyaw, initial_quaternion)
        ).rotation
        (r00, r01, r02), (r10, r11, r12), _ = rotation
        for dy in y_offsets:
            base_y = initial_translation[1] + dy
            for dx in x_offsets:
                evaluated += 1
                base_x = initial_translation[0] + dx
                known = 0
                total = 0
                for px, py, pz in scan:
                    wx = r00 * px + r01 * py + r02 * pz + base_x
                    wy = r10 * px + r11 * py + r12 * pz + base_y
                    ix = math.floor((wx - origin_x) / resolution)
                    iy = math.floor((wy - origin_y) / resolution)
                    if 0 <= ix < width and 0 <= iy < height:
                        cell = data[iy * width + ix]
                        if cell == OCCUPIED:
                            known += 1
                            total += 1
                        elif cell == FREE:
                            known += 1
                            total -= 1
                if known < min_known:
                    continue
                score = total / known
                norm = dx * dx + dy * dy + dyaw * dyaw
                # Greater is better: score, known points, then smaller norm,
                # then ascending dx, dy, dyaw (encoded as negated values).
                key = (score, known, -norm, -dx, -dy, -dyaw)
                if best_key is None or key > best_key:
                    best_key = key
                    best_offsets = (dx, dy, dyaw)
                    best_score = score
                    best_known = known

    if best_offsets is None:
        return CorrelativeScanMatchResult(
            pose=initial_pose,
            matched=False,
            score=None,
            known_points=0,
            evaluated_candidates=evaluated,
        )

    dx, dy, dyaw = best_offsets
    pose = Pose3(
        (
            initial_translation[0] + dx,
            initial_translation[1] + dy,
            initial_translation[2],
        ),
        _yaw_quaternion(dyaw, initial_quaternion),
    )
    return CorrelativeScanMatchResult(
        pose=pose,
        matched=best_score >= score_threshold,
        score=best_score,
        known_points=best_known,
        evaluated_candidates=evaluated,
    )
