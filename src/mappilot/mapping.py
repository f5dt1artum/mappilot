"""Deterministic 2D occupancy-grid mapping from time-ordered laser scans.

The :func:`build_occupancy_grid` entry point consumes scans in time order —
each a unique non-boolean integer id, a world-frame
:class:`~mappilot.geometry.Pose3` sensor pose, and a 3D point cloud in the
frame's local coordinates — and casts one 2D ray per point into an immutable
:class:`OccupancyGrid2D`.

For every frame the points are first transformed into the world frame with
the frame pose and then projected onto the x-y plane. Grid bounds are the
smallest resolution-aligned rectangle covering every sensor origin and every
effective ray endpoint (including range-truncated endpoints), expanded by
``padding`` on each side and aligned outward; the grid origin is the
lower-left outer corner. World coordinates map to cell indices with
``floor((world - origin) / resolution)``.

Each ray walks the standard 2D Bresenham line from the sensor cell to the
hit cell: cells strictly before the endpoint are free, the endpoint is
occupied. Conflicting observations resolve occupied-wins-free; cells no ray
touches stay unknown. With ``max_range`` set, rays whose horizontal length
exceeds the range stop at the range boundary and the truncated endpoint is
free only, never occupied. A valid point with zero horizontal distance
occupies its own cell directly.

The result depends only on the observation set, never on input order: frames
are processed sorted by id, bounds use commutative min/max, and the
occupied-wins merge is order-independent.

This module is pure Python and depends only on the standard library plus
:mod:`mappilot.geometry` and :mod:`mappilot.registration`. Importing it never
starts the HTTP service.

Validation policy mirrors :mod:`mappilot.geometry` and
:mod:`mappilot.registration`: non-iterable scans or clouds, malformed
records, non-integer (including boolean) ids, non-:class:`Pose3` poses,
wrong point dimensions, non-real (including boolean) coordinates or
parameters, and a non-``None`` ``max_range`` of the wrong type raise
:class:`TypeError`; empty scans or clouds, duplicate ids, non-finite
coordinates, a non-positive or non-finite ``resolution``, a negative or
non-finite ``padding``, and a non-positive or non-finite ``max_range`` raise
:class:`ValueError`.
"""

from __future__ import annotations

import math
import numbers
from typing import Iterable, NamedTuple

from .geometry import Pose3
from .registration import _point_cloud, _real_scalar

__all__ = ["OccupancyGrid2D", "build_occupancy_grid"]

# Cell states stored in OccupancyGrid2D.data.
UNKNOWN = -1
FREE = 0
OCCUPIED = 100


class OccupancyGrid2D(NamedTuple):
    """Immutable 2D occupancy grid.

    ``resolution`` is the cell edge length in world units; ``origin`` is the
    world position ``(x, y)`` of the lower-left outer corner; ``width`` and
    ``height`` count cells along x and y. ``data`` is a flat immutable tuple
    ordered with y increasing and x increasing inside each row
    (``index = y * width + x``); values are -1 (unknown), 0 (free), or
    100 (occupied).
    """

    resolution: float
    origin: tuple[float, float]
    width: int
    height: int
    data: tuple[int, ...]


# -- validation ---------------------------------------------------------------


def _resolution(value: object) -> float:
    result = _real_scalar(value, "resolution")
    if result <= 0.0:
        raise ValueError(f"resolution must be greater than zero, got {result}")
    return result


def _padding(value: object) -> float:
    result = _real_scalar(value, "padding")
    if result < 0.0:
        raise ValueError(f"padding must be non-negative, got {result}")
    return result


def _max_range(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(
            f"max_range must be a real number or None, got {type(value).__name__}"
        )
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"max_range must be finite, got {result!r}")
    if result <= 0.0:
        raise ValueError(f"max_range must be greater than zero, got {result}")
    return result


def _parse_scans(
    value: object,
) -> tuple[tuple[int, Pose3, tuple[tuple[float, float, float], ...]], ...]:
    """Validate the ordered scan sequence into ``(id, pose, cloud)`` triples.

    The iterable and every cloud generator are consumed exactly once and the
    input objects are never modified.
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
        raise ValueError("scans must contain at least one frame")

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
                f"{name}[0] must be an integer frame id, "
                f"got {type(frame_id).__name__}"
            )
        frame_id = int(frame_id)

        pose = items[1]
        if not isinstance(pose, Pose3):
            raise TypeError(f"{name}[1] must be a Pose3, got {type(pose).__name__}")

        cloud = _point_cloud(items[2], f"{name}[2]", minimum=1)

        if frame_id in seen_ids:
            raise ValueError(f"duplicate frame id: {frame_id}")
        seen_ids.add(frame_id)
        frames.append((frame_id, pose, cloud))
    return tuple(frames)


# -- rasterization ------------------------------------------------------------


def _bresenham(
    x0: int, y0: int, x1: int, y1: int
) -> tuple[tuple[int, int], ...]:
    """Standard 2D Bresenham line, endpoints included, in traversal order."""
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    error = dx - dy
    cells: list[tuple[int, int]] = []
    while True:
        cells.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        doubled = 2 * error
        if doubled > -dy:
            error -= dy
            x0 += sx
        if doubled < dx:
            error += dx
            y0 += sy
    return tuple(cells)


# -- public entry point -------------------------------------------------------


def build_occupancy_grid(
    scans: Iterable[tuple[int, Pose3, Iterable[Iterable[float]]]],
    resolution: float = 0.05,
    padding: float = 0.0,
    max_range: float | None = None,
) -> OccupancyGrid2D:
    """Build a deterministic 2D occupancy grid from time-ordered laser scans.

    Parameters
    ----------
    scans:
        Time-ordered iterable of ``(id, pose, points)`` records. ``id`` is a
        unique non-boolean integer, ``pose`` the sensor's world-frame
        :class:`~mappilot.geometry.Pose3`, and ``points`` a non-empty
        iterable of finite ``(x, y, z)`` points in the frame's local
        coordinates. The iterable and every point-cloud generator are
        consumed exactly once and the input objects are never modified. The
        result depends only on the observation set, not on frame order.
    resolution:
        Cell edge length in world units, a strictly positive finite number,
        default 0.05.
    padding:
        Extra world distance added on every side of the tight bounds before
        outward alignment; a non-negative finite number, default 0.0.
    max_range:
        Optional horizontal sensor range. Rays longer than it are truncated
        at the range boundary and mark free space only; ``None`` (default)
        keeps every hit as an occupied endpoint.

    Returns
    -------
    OccupancyGrid2D
        Immutable grid whose flat ``data`` is ordered by increasing y and
        increasing x within each row. Cells are -1 unknown, 0 free, and 100
        occupied; occupied observations always override conflicting free
        observations.
    """
    cell_size = _resolution(resolution)
    border = _padding(padding)
    range_limit = _max_range(max_range)

    frames = _parse_scans(scans)
    # Sorting by id makes processing independent of the input ordering.
    frames = tuple(sorted(frames, key=lambda frame: frame[0]))

    # First pass: project every ray into the world frame, applying the range
    # truncation, while collecting the tight world bounds.
    # Each ray is (sensor_x, sensor_y, end_x, end_y, occupied_endpoint).
    rays: list[tuple[float, float, float, float, bool]] = []
    min_x = math.inf
    min_y = math.inf
    max_x = -math.inf
    max_y = -math.inf

    for _frame_id, pose, cloud in frames:
        ox, oy, _oz = pose.translation
        rotation = pose.rotation
        r0, r1, _r2 = rotation
        min_x = min(min_x, ox)
        min_y = min(min_y, oy)
        max_x = max(max_x, ox)
        max_y = max(max_y, oy)
        for px, py, pz in cloud:
            wx = r0[0] * px + r0[1] * py + r0[2] * pz + ox
            wy = r1[0] * px + r1[1] * py + r1[2] * pz + oy
            occupied_endpoint = True
            if range_limit is not None:
                distance = math.hypot(wx - ox, wy - oy)
                if distance > range_limit:
                    scale = range_limit / distance
                    wx = ox + (wx - ox) * scale
                    wy = oy + (wy - oy) * scale
                    occupied_endpoint = False
            rays.append((ox, oy, wx, wy, occupied_endpoint))
            min_x = min(min_x, wx)
            min_y = min(min_y, wy)
            max_x = max(max_x, wx)
            max_y = max(max_y, wy)

    # Outward alignment: the outer edges lie on resolution multiples.
    low_cell_x = math.floor((min_x - border) / cell_size)
    low_cell_y = math.floor((min_y - border) / cell_size)
    high_cell_x = math.floor((max_x + border) / cell_size)
    high_cell_y = math.floor((max_y + border) / cell_size)
    origin_x = low_cell_x * cell_size
    origin_y = low_cell_y * cell_size
    width = high_cell_x - low_cell_x + 1
    height = high_cell_y - low_cell_y + 1

    cells = [UNKNOWN] * (width * height)

    def cell_indices(coord_x: float, coord_y: float) -> tuple[int, int]:
        # Indices follow floor((world - origin) / resolution); the clamp
        # absorbs the sub-ULP rounding that can put a boundary point one cell
        # outside the outward-aligned rectangle.
        ix = math.floor((coord_x - origin_x) / cell_size)
        iy = math.floor((coord_y - origin_y) / cell_size)
        if ix < 0:
            ix = 0
        elif ix >= width:
            ix = width - 1
        if iy < 0:
            iy = 0
        elif iy >= height:
            iy = height - 1
        return ix, iy

    # Second pass: Bresenham traversal with occupied-wins-free merging.
    for ox, oy, wx, wy, occupied_endpoint in rays:
        x0, y0 = cell_indices(ox, oy)
        x1, y1 = cell_indices(wx, wy)
        line = _bresenham(x0, y0, x1, y1)
        if occupied_endpoint:
            for x, y in line[:-1]:
                index = y * width + x
                if cells[index] == UNKNOWN:
                    cells[index] = FREE
            ex, ey = line[-1]
            cells[ey * width + ex] = OCCUPIED
        else:
            for x, y in line:
                index = y * width + x
                if cells[index] == UNKNOWN:
                    cells[index] = FREE

    return OccupancyGrid2D(
        resolution=cell_size,
        origin=(float(origin_x), float(origin_y)),
        width=width,
        height=height,
        data=tuple(cells),
    )
