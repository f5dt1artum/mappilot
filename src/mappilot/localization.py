"""Deterministic correlative scan matching on a 2-D occupancy grid.

The :func:`correlative_scan_match` entry point relocalizes a local 3D scan
against a prebuilt :class:`~mappilot.mapping.OccupancyGrid2D` when a rough
prior is available, e.g. for recovery after tracking loss. It exhaustively
evaluates a deterministic lattice of candidate poses around the initial
guess:

* Candidate offsets are integer multiples of the per-axis step whose
  absolute value does not exceed the matching window; zero is always
  included, so the initial pose itself is always evaluated.
* A candidate adds ``dx``/``dy`` directly to the initial world translation
  (z is kept unchanged) and left-composes a rotation of ``dyaw`` about the
  world z axis onto the initial rotation.
* Each candidate transforms the scan with the usual
  :class:`~mappilot.geometry.Pose3` semantics (``p_world = R * p_local + t``)
  and projects every transformed point into the grid with the
  ``floor((p - origin) / resolution)`` rule; the z coordinate never takes
  part in scoring.
* A point landing on an occupied cell contributes +1, on a free cell -1;
  unknown cells and points outside the map contribute nothing and are not
  counted as known. Candidates with at least ``min_known_points`` known
  points score the mean of their contributions.

The best candidate wins by highest score, then by most known points, then
by the smallest ``dx**2 + dy**2 + dyaw**2``, and finally by ascending
``dx``, ``dy``, ``dyaw`` — so the result is fully deterministic and
identical inputs produce value-identical results. The best pose is
returned even when its score stays below ``min_score`` (``matched=False``);
only when no candidate reaches ``min_known_points`` is the original
initial pose returned with ``score=None`` and ``known_points=0``.
``evaluated_candidates`` always counts every candidate actually searched.

This module is pure Python and depends only on the standard library plus
:mod:`mappilot.geometry` and :mod:`mappilot.mapping`. Importing it never
starts the HTTP service.

Validation policy mirrors :mod:`mappilot.geometry` and
:mod:`mappilot.mapping`: a non-iterable scan, wrong point dimensions,
non-real (including boolean) coordinates or real-valued parameters, a
non-integer (including boolean) ``min_known_points``, a grid that is not an
:class:`~mappilot.mapping.OccupancyGrid2D`, or an initial pose that is not
a :class:`~mappilot.geometry.Pose3` raise :class:`TypeError`; an empty
scan, non-finite coordinates, negative or non-finite windows, non-positive
or non-finite steps, ``min_known_points`` below one, and a ``min_score``
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
    only when the best score reaches ``min_score``; ``score`` is the best
    candidate's mean cell contribution (``None`` when no candidate
    qualified); ``known_points`` is the best candidate's known-point count
    (0 when no candidate qualified); ``evaluated_candidates`` is the total
    number of candidates actually searched.
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


def _point_cloud(value: object, name: str) -> tuple[tuple[float, float, float], ...]:
    """Validate a non-empty cloud of 3D finite points."""
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be an iterable of 3D points")
    try:
        raw_points = tuple(value)  # type: ignore[call-overload]
    except TypeError:
        raise TypeError(f"{name} must be an iterable of 3D points") from None
    if len(raw_points) == 0:
        raise ValueError(f"{name} must contain at least one point, got 0")

    points: list[tuple[float, float, float]] = []
    for index, raw_point in enumerate(raw_points):
        if isinstance(raw_point, (str, bytes, bytearray)):
            raise TypeError(f"{name}[{index}] must be a sequence of 3 real numbers")
        try:
            raw_coords = tuple(raw_point)  # type: ignore[call-overload]
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


def _min_known(value: object) -> int:
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
    if result < -1.0 or result > 1.0:
        raise ValueError(f"min_score must lie in [-1, 1], got {result}")
    return result


# -- candidate lattice --------------------------------------------------------


def _offsets(window: float, step: float) -> tuple[float, ...]:
    """Integer step multiples with absolute value at most ``window``; includes zero."""
    count = int(math.floor(window / step))
    return tuple(k * step for k in range(-count, count + 1))


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


# -- public entry point -------------------------------------------------------


def correlative_scan_match(
    occupancy_grid: OccupancyGrid2D,
    scan_points: Iterable[Iterable[float]],
    initial_pose: Pose3,
    x_window: float,
    x_step: float,
    y_window: float,
    y_step: float,
    yaw_window: float,
    yaw_step: float,
    min_known_points: int,
    min_score: float,
) -> CorrelativeScanMatchResult:
    """Match a local scan against an occupancy grid around a rough prior.

    Parameters
    ----------
    occupancy_grid:
        The :class:`~mappilot.mapping.OccupancyGrid2D` to match against.
    scan_points:
        Non-empty iterable of finite ``(x, y, z)`` points in the sensor's
        local frame. The iterable is consumed exactly once and never
        modified; z never takes part in scoring.
    initial_pose:
        Prior world-frame :class:`~mappilot.geometry.Pose3` the search is
        centered on.
    x_window, x_step, y_window, y_step, yaw_window, yaw_step:
        Per-axis search window (non-negative finite) and step (positive
        finite). Candidate offsets are integer multiples of the step whose
        absolute value does not exceed the window; zero is always included.
        A candidate adds ``dx``/``dy`` to the initial world translation
        (z unchanged) and left-composes a ``dyaw`` rotation about the world
        z axis onto the initial rotation.
    min_known_points:
        Minimum number of points landing on known (occupied or free) cells
        for a candidate to be scored; a non-boolean integer, at least 1.
    min_score:
        Score threshold for ``matched=True``; a finite real in ``[-1, 1]``.

    Returns
    -------
    CorrelativeScanMatchResult
        The highest-scoring candidate. Ties resolve to more known points,
        then smaller ``dx**2 + dy**2 + dyaw**2``, then ascending ``dx``,
        ``dy``, ``dyaw``. A best score below ``min_score`` still returns the
        best pose and score with ``matched=False``. When no candidate
        reaches ``min_known_points``, returns the original initial pose
        with ``matched=False``, ``score=None`` and ``known_points=0``.
        ``evaluated_candidates`` always counts every candidate searched.
    """
    if not isinstance(occupancy_grid, OccupancyGrid2D):
        raise TypeError(
            "occupancy_grid must be an OccupancyGrid2D, "
            f"got {type(occupancy_grid).__name__}"
        )
    if not isinstance(initial_pose, Pose3):
        raise TypeError(
            f"initial_pose must be a Pose3, got {type(initial_pose).__name__}"
        )
    scan = _point_cloud(scan_points, "scan_points")

    x_win = _window(x_window, "x_window")
    x_st = _step(x_step, "x_step")
    y_win = _window(y_window, "y_window")
    y_st = _step(y_step, "y_step")
    yaw_win = _window(yaw_window, "yaw_window")
    yaw_st = _step(yaw_step, "yaw_step")
    known_minimum = _min_known(min_known_points)
    score_threshold = _min_score(min_score)

    x_offsets = _offsets(x_win, x_st)
    y_offsets = _offsets(y_win, y_st)
    yaw_offsets = _offsets(yaw_win, yaw_st)
    evaluated = len(x_offsets) * len(y_offsets) * len(yaw_offsets)

    origin_x, origin_y = occupancy_grid.origin
    resolution = occupancy_grid.resolution
    width = occupancy_grid.width
    height = occupancy_grid.height
    data = occupancy_grid.data
    tx, ty, tz = initial_pose.translation
    base_quaternion = initial_pose.quaternion

    best_key: tuple[float, int, float, float, float, float] | None = None
    best_pose: Pose3 | None = None
    best_score = 0.0
    best_known = 0

    for dx in x_offsets:
        for dy in y_offsets:
            for dyaw in yaw_offsets:
                half = 0.5 * dyaw
                quaternion = _quaternion_multiply(
                    (math.cos(half), 0.0, 0.0, math.sin(half)), base_quaternion
                )
                pose = Pose3((tx + dx, ty + dy, tz), quaternion)
                known = 0
                total = 0
                for point in scan:
                    wx, wy, _wz = pose.transform_point(point)
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
                if known < known_minimum:
                    continue
                score = total / known
                key = (-score, -known, dx * dx + dy * dy + dyaw * dyaw, dx, dy, dyaw)
                if best_key is None or key < best_key:
                    best_key = key
                    best_pose = pose
                    best_score = score
                    best_known = known

    if best_pose is None:
        return CorrelativeScanMatchResult(initial_pose, False, None, 0, evaluated)
    return CorrelativeScanMatchResult(
        best_pose, best_score >= score_threshold, best_score, best_known, evaluated
    )
