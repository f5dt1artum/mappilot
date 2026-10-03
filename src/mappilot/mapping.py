"""Deterministic 2D occupancy-grid mapping from time-ordered laser scans.

The :func:`build_occupancy_grid` entry point takes a time-ordered sequence of
scans — each a unique non-boolean integer id, the sensor world-frame
:class:`~mappilot.geometry.Pose3`, and a non-empty sequence of 3D points in
that frame's local coordinates — and casts every sensor-to-return ray into a
2D occupancy grid:

1. Each point is transformed into the world frame with the frame's
   :class:`~mappilot.geometry.Pose3` and projected onto the x-y plane (the
   z coordinate is otherwise ignored).
2. The grid bounds are the smallest resolution-aligned rectangle containing
   every sensor origin and every valid ray endpoint, enlarged by ``padding``
   on all four sides and aligned outward. ``origin`` is the lower-left outer
   corner; a world point maps to cell ``floor((p - origin) / resolution)``.
3. Each ray walks the standard 2-D Bresenham line from the sensor cell to
   the hit cell: cells strictly before the endpoint become free, the
   endpoint cell occupied. Conflicting observations resolve occupied-over-
   free; cells never touched stay unknown.
4. When ``max_range`` is set, rays whose horizontal length exceeds it are
   clipped to the range boundary: the clipped endpoint (and the ray leading
   to it) is free only and never produces an occupied hit. A valid point at
   zero horizontal distance occupies its own cell directly.

The result is independent of input order for the same observation set.

This module is pure Python and depends only on the standard library plus
:mod:`mappilot.geometry`. Importing it never starts the HTTP service.

Validation policy mirrors :mod:`mappilot.geometry` and
:mod:`mappilot.registration`: non-iterable scans or clouds, malformed
records, non-integer (including boolean) ids, non-:class:`Pose3` poses,
wrong point dimensions, non-real (including boolean) coordinates or
parameters raise :class:`TypeError`; duplicate ids, non-finite coordinates,
an empty scan sequence or an empty frame/cloud, a non-positive or
non-finite ``resolution``, a negative or non-finite ``padding``, and a
non-positive or non-finite ``max_range`` raise :class:`ValueError`.
"""

from __future__ import annotations

import math
import numbers
from typing import Iterable, NamedTuple

from .geometry import Pose3

__all__ = ["OccupancyGrid2D", "build_occupancy_grid"]

# Cell states encoded in ``OccupancyGrid2D.data``.
UNKNOWN = -1
FREE = 0
OCCUPIED = 100


class OccupancyGrid2D(NamedTuple):
    """Immutable 2-D occupancy grid.

    ``origin`` is the lower-left outer corner in world coordinates
    ``(x, y)``; cell ``(ix, iy)`` spans
    ``[origin_x + ix*resolution, origin_x + (ix+1)*resolution)`` in x and
    the matching interval in y. ``data`` is a flat tuple with y increasing
    and x increasing within each row (index ``iy * width + ix``); entries
    are :data:`UNKNOWN` (-1), :data:`FREE` (0), or :data:`OCCUPIED` (100).
    """

    resolution: float
    origin: tuple[float, float]
    width: int
    height: int
    data: tuple[int, ...]


# -- validation -------------------------------------------------------------


def _real_scalar(value: object, name: str) -> float:
    """Coerce one real, finite, non-boolean scalar to float."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {result!r}")
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


def _parse_scans(
    value: object,
) -> tuple[tuple[int, Pose3, tuple[tuple[float, float, float], ...]], ...]:
    """Validate the ordered scan sequence into ``(id, pose, cloud)`` triples.

    The iterable and every cloud are consumed exactly once and never
    modified.
    """
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError("scans must be an iterable of (id, pose, points) records")
    try:
        raw_frames = tuple(value)  # type: ignore[call-overload]
    except TypeError:
        raise TypeError(
            "scans must be an iterable of (id, pose, points) records"
        ) from None
    if len(raw_frames) == 0:
        raise ValueError("scans must contain at least one frame, got 0")

    frames: list[tuple[int, Pose3, tuple[tuple[float, float, float], ...]]] = []
    seen_ids: set[int] = set()
    for position, raw_frame in enumerate(raw_frames):
        name = f"scans[{position}]"
        if isinstance(raw_frame, (str, bytes, bytearray)):
            raise TypeError(f"{name} must be a sequence of 3 elements")
        try:
            items = tuple(raw_frame)  # type: ignore[call-overload]
        except TypeError:
            raise TypeError(f"{name} must be a sequence of 3 elements") from None
        if len(items) != 3:
            raise TypeError(f"{name} must have exactly 3 elements, got {len(items)}")

        frame_id = items[0]
        if isinstance(frame_id, bool) or not isinstance(frame_id, numbers.Integral):
            raise TypeError(
                f"{name}[0] must be an integer scan id, "
                f"got {type(frame_id).__name__}"
            )
        frame_id = int(frame_id)

        pose = items[1]
        if not isinstance(pose, Pose3):
            raise TypeError(f"{name}[1] must be a Pose3, got {type(pose).__name__}")

        cloud = _point_cloud(items[2], f"{name}[2]")

        if frame_id in seen_ids:
            raise ValueError(f"duplicate scan id: {frame_id}")
        seen_ids.add(frame_id)
        frames.append((frame_id, pose, cloud))
    return tuple(frames)


# -- rasterization ----------------------------------------------------------


def _bresenham(
    x0: int, y0: int, x1: int, y1: int
) -> tuple[tuple[int, int], ...]:
    """Standard 2-D Bresenham walk from one cell to another, endpoints included."""
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    step_x = 1 if x0 < x1 else -1
    step_y = 1 if y0 < y1 else -1
    error = dx - dy
    cells: list[tuple[int, int]] = []
    x, y = x0, y0
    while True:
        cells.append((x, y))
        if x == x1 and y == y1:
            break
        doubled = 2 * error
        if doubled > -dy:
            error -= dy
            x += step_x
        if doubled < dx:
            error += dx
            y += step_y
    return tuple(cells)


# -- public entry point -------------------------------------------------------


def build_occupancy_grid(
    scans: Iterable[tuple[int, Pose3, Iterable[Iterable[float]]]],
    resolution: float = 0.05,
    padding: float = 0.0,
    max_range: float | None = None,
) -> OccupancyGrid2D:
    """Build a deterministic 2-D occupancy grid from time-ordered laser scans.

    Parameters
    ----------
    scans:
        Time-ordered iterable of ``(id, pose, points)`` records. ``id`` is a
        unique non-boolean integer, ``pose`` the sensor world-frame
        :class:`~mappilot.geometry.Pose3`, and ``points`` a non-empty
        iterable of finite ``(x, y, z)`` points in the frame's local
        coordinates. The iterable and every cloud are consumed exactly once
        and never modified.
    resolution:
        Grid cell edge length in world units; a strictly positive finite
        number, default 0.05.
    padding:
        Extra world distance added on all four sides before the bounds are
        aligned outward; a non-negative finite number, default 0.0.
    max_range:
        When set (strictly positive finite number), rays whose horizontal
        length exceeds it are clipped to the range boundary and mark cells
        free only. ``None`` (default) keeps every return as a hit.

    Returns
    -------
    OccupancyGrid2D
        Immutable grid. Cells are :data:`OccupancyGrid2D` constants
        ``-1``/``0``/``100`` for unknown/free/occupied. Reordering the
        input frames or points never changes the grid built from the same
        observation set.
    """
    frames = _parse_scans(scans)

    cell_size = _real_scalar(resolution, "resolution")
    if cell_size <= 0.0:
        raise ValueError(f"resolution must be greater than zero, got {cell_size}")
    margin = _real_scalar(padding, "padding")
    if margin < 0.0:
        raise ValueError(f"padding must be non-negative, got {margin}")
    range_limit: float | None = None
    if max_range is not None:
        range_limit = _real_scalar(max_range, "max_range")
        if range_limit <= 0.0:
            raise ValueError(f"max_range must be greater than zero, got {range_limit}")

    # Gather the 2-D endpoints of every ray (sensor origin plus the actual
    # ray endpoint used: hit, clipped range boundary, or the origin itself
    # for zero-distance returns) and the per-ray cell walk, expressed in
    # world coordinates until the bounds are known.
    #
    # Each ray is (sensor_xy, endpoint_xy, occupied_endpoint): clipped rays
    # carry ``occupied_endpoint=False`` so their endpoint stays free.
    origins: list[tuple[float, float]] = []
    rays: list[tuple[tuple[float, float], tuple[float, float], bool]] = []
    for _frame_id, pose, cloud in frames:
        sx, sy, _sz = pose.translation
        sensor = (sx, sy)
        origins.append(sensor)
        for local_point in cloud:
            wx, wy, _wz = pose.transform_point(local_point)
            distance = math.hypot(wx - sx, wy - sy)
            if distance == 0.0:
                # Zero horizontal distance: the point's own cell is occupied
                # directly; there is no ray to walk.
                rays.append((sensor, sensor, True))
            elif range_limit is not None and distance > range_limit:
                scale = range_limit / distance
                clipped = (sx + (wx - sx) * scale, sy + (wy - sy) * scale)
                rays.append((sensor, clipped, False))
            else:
                rays.append((sensor, (wx, wy), True))

    endpoints = origins + [endpoint for _origin, endpoint, _hit in rays]
    min_x = min(point[0] for point in endpoints)
    max_x = max(point[0] for point in endpoints)
    min_y = min(point[1] for point in endpoints)
    max_y = max(point[1] for point in endpoints)

    # Outward alignment to resolution lines measured from the world origin:
    # integer floor indices make the box exactly resolution-aligned without
    # accumulating floating-point drift. The upper edge uses ``floor(...) +
    # 1`` (rather than ``ceil``) so a point sitting exactly on a grid line
    # still falls inside a cell under the floor index rule.
    left_line = math.floor((min_x - margin) / cell_size)
    bottom_line = math.floor((min_y - margin) / cell_size)
    right_line = math.floor((max_x + margin) / cell_size) + 1
    top_line = math.floor((max_y + margin) / cell_size) + 1
    width = right_line - left_line
    height = top_line - bottom_line
    origin_x = left_line * cell_size
    origin_y = bottom_line * cell_size

    def cell_index(x: float, y: float) -> tuple[int, int]:
        ix = math.floor((x - origin_x) / cell_size)
        iy = math.floor((y - origin_y) / cell_size)
        # Clamp only defends against sub-ULP rounding at an aligned border;
        # every endpoint lies inside the box by construction.
        return max(0, min(width - 1, ix)), max(0, min(height - 1, iy))

    occupied: set[tuple[int, int]] = set()
    freed: set[tuple[int, int]] = set()
    for sensor, endpoint, is_hit in rays:
        x0, y0 = cell_index(*sensor)
        x1, y1 = cell_index(*endpoint)
        if x0 == x1 and y0 == y1:
            if is_hit:
                occupied.add((x0, y0))
            else:
                freed.add((x0, y0))
            continue
        line = _bresenham(x0, y0, x1, y1)
        if is_hit:
            for cell in line[:-1]:
                freed.add(cell)
            occupied.add(line[-1])
        else:
            for cell in line:
                freed.add(cell)

    data = tuple(
        OCCUPIED
        if (ix, iy) in occupied
        else FREE
        if (ix, iy) in freed
        else UNKNOWN
        for iy in range(height)
        for ix in range(width)
    )
    return OccupancyGrid2D(
        resolution=cell_size,
        origin=(origin_x, origin_y),
        width=width,
        height=height,
        data=data,
    )
