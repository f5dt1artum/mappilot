import math
import unittest

from mappilot.geometry import Pose3
from mappilot.mapping import (
    FREE,
    OCCUPIED,
    UNKNOWN,
    OccupancyGrid2D,
    build_occupancy_grid,
)


def row(grid, iy):
    return list(grid.data[iy * grid.width : (iy + 1) * grid.width])


def grid_rows(grid):
    return [row(grid, iy) for iy in range(grid.height)]


class BasicGridTest(unittest.TestCase):
    def test_axis_aligned_ray(self):
        # Sensor at the origin, hit at 1.0 m, resolution 0.25. The hit lies
        # exactly on a grid line, so the cell covering it extends the grid
        # one cell past the line.
        grid = build_occupancy_grid([(0, Pose3(), [(1.0, 0.0, 0.0)])], resolution=0.25)
        self.assertEqual(grid.resolution, 0.25)
        self.assertEqual(grid.origin, (0.0, 0.0))
        self.assertEqual((grid.width, grid.height), (5, 1))
        self.assertEqual(list(grid.data), [FREE, FREE, FREE, FREE, OCCUPIED])

    def test_data_is_flat_row_major_y_increasing(self):
        # Diagonal hit at (3.5, 3.5): cells (0,0),(1,1),(2,2) free,
        # (3,3) occupied, laid out y-row by y-row, x increasing.
        grid = build_occupancy_grid([(0, Pose3(), [(3.5, 3.5, 0.0)])], resolution=1.0)
        self.assertEqual((grid.width, grid.height), (4, 4))
        self.assertEqual(
            grid_rows(grid),
            [
                [FREE, UNKNOWN, UNKNOWN, UNKNOWN],
                [UNKNOWN, FREE, UNKNOWN, UNKNOWN],
                [UNKNOWN, UNKNOWN, FREE, UNKNOWN],
                [UNKNOWN, UNKNOWN, UNKNOWN, OCCUPIED],
            ],
        )
        for iy in range(grid.height):
            for ix in range(grid.width):
                self.assertEqual(grid.data[iy * grid.width + ix], grid_rows(grid)[iy][ix])

    def test_unobserved_cells_stay_unknown(self):
        grid = build_occupancy_grid([(0, Pose3(), [(1.0, 1.0, 0.0)])], resolution=1.0)
        # Bounds cover the 2x2 block from (0,0) to (1,1); only the two
        # Bresenham cells are observed.
        observed = {value for value in grid.data if value != UNKNOWN}
        self.assertEqual(observed, {FREE, OCCUPIED})
        self.assertIn(UNKNOWN, grid.data)

    def test_z_coordinate_ignored_in_projection(self):
        above = build_occupancy_grid([(0, Pose3(), [(1.0, 0.0, 5.0)])], resolution=0.5)
        below = build_occupancy_grid([(0, Pose3(), [(1.0, 0.0, -7.0)])], resolution=0.5)
        self.assertEqual(above, below)

    def test_hit_inside_sensor_cell_occupies_sensor_cell(self):
        grid = build_occupancy_grid([(0, Pose3(), [(0.1, 0.0, 0.0)])], resolution=1.0)
        self.assertEqual((grid.width, grid.height), (1, 1))
        self.assertEqual(list(grid.data), [OCCUPIED])

    def test_zero_horizontal_distance_occupies_cell_directly(self):
        # The return is directly overhead (5 m up): no horizontal ray, the
        # point's own cell is occupied.
        pose = Pose3(translation=(2.0, -3.0, 1.0))
        grid = build_occupancy_grid([(7, pose, [(0.0, 0.0, 5.0)])], resolution=1.0)
        self.assertEqual(grid.origin, (2.0, -3.0))
        self.assertEqual((grid.width, grid.height), (1, 1))
        self.assertEqual(list(grid.data), [OCCUPIED])


class BoundsTest(unittest.TestCase):
    def test_bounds_contain_sensor_origin_and_hit(self):
        pose = Pose3(translation=(10.0, 0.0, 0.0))
        grid = build_occupancy_grid([(0, pose, [(1.0, 0.0, 0.0)])], resolution=1.0)
        # World span 10..11 maps to grid columns 0..1; origin is the
        # lower-left outer corner.
        self.assertEqual(grid.origin, (10.0, 0.0))
        self.assertEqual((grid.width, grid.height), (2, 1))
        self.assertEqual(list(grid.data), [FREE, OCCUPIED])

    def test_padding_expands_and_aligns_outward(self):
        # Span 0..1 with 0.1 m padding at resolution 0.5:
        # floor(-0.2) = -1 line, ceil(2.2) = 3 lines -> origin -0.5, width 4;
        # y: floor(-0.2) = -1, ceil(0.2) = 1 -> origin -0.5, height 2.
        grid = build_occupancy_grid(
            [(0, Pose3(), [(1.0, 0.0, 0.0)])], resolution=0.5, padding=0.1
        )
        self.assertEqual(grid.origin, (-0.5, -0.5))
        self.assertEqual((grid.width, grid.height), (4, 2))
        self.assertEqual(
            grid_rows(grid),
            [
                [UNKNOWN, UNKNOWN, UNKNOWN, UNKNOWN],
                [UNKNOWN, FREE, FREE, OCCUPIED],
            ],
        )

    def test_negative_world_coordinates(self):
        pose = Pose3(translation=(-1.0, -1.0, 0.0))
        grid = build_occupancy_grid([(0, pose, [(1.5, 1.5, 0.0)])], resolution=1.0)
        self.assertEqual(grid.origin, (-1.0, -1.0))
        self.assertEqual((grid.width, grid.height), (2, 2))
        self.assertEqual(
            grid_rows(grid),
            [[FREE, UNKNOWN], [UNKNOWN, OCCUPIED]],
        )

    def test_grid_always_at_least_one_by_one(self):
        # Zero-distance return at a single point: degenerate span still
        # yields a 1x1 grid.
        grid = build_occupancy_grid([(0, Pose3(), [(0.0, 0.0, 0.0)])], resolution=0.05)
        self.assertEqual((grid.width, grid.height), (1, 1))
        self.assertEqual(list(grid.data), [OCCUPIED])


class PoseTransformTest(unittest.TestCase):
    def test_translated_frame(self):
        pose = Pose3(translation=(10.0, 0.0, 0.0))
        grid = build_occupancy_grid([(0, pose, [(1.0, 0.0, 0.0)])], resolution=1.0)
        self.assertEqual(list(grid.data), [FREE, OCCUPIED])

    def test_rotated_frame_projects_ray_to_world_xy(self):
        # 90 degree yaw: local +x maps to world +y.
        yaw_90 = (math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4))
        pose = Pose3(translation=(0.0, 0.0, 0.0), quaternion=yaw_90)
        grid = build_occupancy_grid([(0, pose, [(2.5, 0.0, 0.0)])], resolution=1.0)
        # Cells along world +y: (0,0) free ... (0,2) occupied.
        self.assertEqual(grid.width, 1)
        self.assertEqual(
            [grid.data[iy] for iy in range(grid.height)],
            [FREE, FREE, OCCUPIED],
        )


class MaxRangeTest(unittest.TestCase):
    def test_hit_exactly_at_range_is_occupied(self):
        grid = build_occupancy_grid(
            [(0, Pose3(), [(1.0, 0.0, 0.0)])], resolution=0.5, max_range=1.0
        )
        self.assertEqual(list(grid.data), [FREE, FREE, OCCUPIED])

    def test_ray_beyond_range_is_clipped_free_only(self):
        # Return at 3 m, range 1 m: the ray up to the 1 m boundary is free,
        # including the clipped endpoint cell; nothing is occupied.
        grid = build_occupancy_grid(
            [(0, Pose3(), [(3.0, 0.0, 0.0)])], resolution=1.0, max_range=1.0
        )
        self.assertEqual(grid.origin, (0.0, 0.0))
        self.assertTrue(all(value == FREE for value in grid.data))
        self.assertNotIn(OCCUPIED, grid.data)

    def test_clipped_ray_marks_all_cells_free(self):
        grid = build_occupancy_grid(
            [(0, Pose3(), [(5.0, 0.0, 0.0)])], resolution=1.0, max_range=3.5
        )
        # Endpoints are the sensor (0) and the clip point (3.5): 4 free cells.
        self.assertEqual((grid.width, grid.height), (4, 1))
        self.assertEqual(list(grid.data), [FREE, FREE, FREE, FREE])

    def test_none_keeps_every_hit(self):
        grid = build_occupancy_grid(
            [(0, Pose3(), [(100.0, 0.0, 0.0)])], resolution=1.0, max_range=None
        )
        self.assertIn(OCCUPIED, grid.data)


class FusionTest(unittest.TestCase):
    def _scans(self):
        # First ray frees cells along y = 0; the second ray ends on a cell
        # (1, 0) that the first ray had marked free.
        return [
            (0, Pose3(), [(3.5, 0.0, 0.0)]),
            (1, Pose3(translation=(0.0, 1.0, 0.0)), [(1.5, -1.0, 0.0)]),
        ]

    def test_occupied_wins_over_free(self):
        grid = build_occupancy_grid(self._scans(), resolution=1.0)
        self.assertEqual(grid.data[0 * grid.width + 1], OCCUPIED)

    def test_result_is_order_independent(self):
        first = build_occupancy_grid(self._scans(), resolution=1.0)
        reversed_scans = build_occupancy_grid(list(reversed(self._scans())), resolution=1.0)
        self.assertEqual(first, reversed_scans)

    def test_permuted_frames_and_points_are_identical(self):
        scans = [
            (3, Pose3(translation=(2.0, 0.0, 0.0)), [(0.5, 0.0, 0.0), (0.2, 0.3, 0.0)]),
            (1, Pose3(), [(1.5, 0.0, 0.0), (1.2, 0.4, 0.0)]),
            (2, Pose3(translation=(0.0, 2.0, 0.0)), [(0.1, -1.0, 0.0)]),
        ]

        def permuted():
            yield scans[2]
            frame = scans[0]
            yield (frame[0], frame[1], reversed(list(frame[2])))
            frame = scans[1]
            yield (frame[0], frame[1], list(reversed(frame[2])))

        direct = build_occupancy_grid(scans, resolution=0.5)
        shuffled = build_occupancy_grid(permuted(), resolution=0.5)
        self.assertEqual(direct, shuffled)


class ConsumptionTest(unittest.TestCase):
    def test_scan_and_cloud_generators_consumed_once(self):
        consumed = {"scans": 0, "clouds": 0}

        def cloud():
            consumed["clouds"] += 1
            yield (1.0, 0.0, 0.0)
            yield (0.5, 0.2, 0.0)

        def scans():
            consumed["scans"] += 1
            yield (0, Pose3(), cloud())

        grid = build_occupancy_grid(scans(), resolution=0.5)
        self.assertEqual(consumed, {"scans": 1, "clouds": 1})
        self.assertIn(OCCUPIED, grid.data)

    def test_inputs_are_not_modified(self):
        points = [[1.0, 0.0, 0.0], [0.5, 0.1, 0.0]]
        points_copy = [list(point) for point in points]
        scans = [(0, Pose3(), points)]
        build_occupancy_grid(scans, resolution=0.5)
        self.assertEqual(points, points_copy)
        self.assertEqual(len(scans), 1)

    def test_defaults(self):
        grid = build_occupancy_grid([(0, Pose3(), [(1.0, 0.0, 0.0)])])
        self.assertEqual(grid.resolution, 0.05)
        # Hit on the 1.0 grid line needs one cell beyond it.
        self.assertEqual(grid.width, 21)


class ImmutabilityTest(unittest.TestCase):
    def test_grid_is_immutable(self):
        grid = build_occupancy_grid([(0, Pose3(), [(1.0, 0.0, 0.0)])], resolution=0.5)
        self.assertIsInstance(grid, OccupancyGrid2D)
        self.assertIsInstance(grid.data, tuple)
        self.assertIsInstance(grid.origin, tuple)
        with self.assertRaises(AttributeError):
            grid.resolution = 1.0  # type: ignore[misc]
        with self.assertRaises(TypeError):
            grid.data[0] = OCCUPIED  # type: ignore[index]


class TypeErrorTest(unittest.TestCase):
    def test_non_iterable_scans(self):
        with self.assertRaises(TypeError):
            build_occupancy_grid(42)
        with self.assertRaises(TypeError):
            build_occupancy_grid("not-scans")

    def test_bad_frame_shape(self):
        with self.assertRaises(TypeError):
            build_occupancy_grid([(0, Pose3())])
        with self.assertRaises(TypeError):
            build_occupancy_grid([(0, Pose3(), [(1.0, 0.0, 0.0)], "extra")])
        with self.assertRaises(TypeError):
            build_occupancy_grid([7])

    def test_id_must_be_non_bool_integer(self):
        with self.assertRaises(TypeError):
            build_occupancy_grid([("0", Pose3(), [(1.0, 0.0, 0.0)])])
        with self.assertRaises(TypeError):
            build_occupancy_grid([(True, Pose3(), [(1.0, 0.0, 0.0)])])
        with self.assertRaises(TypeError):
            build_occupancy_grid([(1.5, Pose3(), [(1.0, 0.0, 0.0)])])

    def test_pose_must_be_pose3(self):
        with self.assertRaises(TypeError):
            build_occupancy_grid([(0, ((1.0, 0.0, 0.0),), [(1.0, 0.0, 0.0)])])
        with self.assertRaises(TypeError):
            build_occupancy_grid([(0, "pose", [(1.0, 0.0, 0.0)])])

    def test_cloud_must_be_iterable(self):
        with self.assertRaises(TypeError):
            build_occupancy_grid([(0, Pose3(), 7)])

    def test_point_dimensions(self):
        with self.assertRaises(TypeError):
            build_occupancy_grid([(0, Pose3(), [(1.0, 0.0)])])
        with self.assertRaises(TypeError):
            build_occupancy_grid([(0, Pose3(), [(1.0, 0.0, 0.0, 0.0)])])
        with self.assertRaises(TypeError):
            build_occupancy_grid([(0, Pose3(), [7])])

    def test_coordinates_must_be_non_bool_real(self):
        with self.assertRaises(TypeError):
            build_occupancy_grid([(0, Pose3(), [("1", 0.0, 0.0)])])
        with self.assertRaises(TypeError):
            build_occupancy_grid([(0, Pose3(), [(True, 0.0, 0.0)])])
        with self.assertRaises(TypeError):
            build_occupancy_grid([(0, Pose3(), [(complex(1), 0.0, 0.0)])])

    def test_parameter_types(self):
        scans = [(0, Pose3(), [(1.0, 0.0, 0.0)])]
        with self.assertRaises(TypeError):
            build_occupancy_grid(scans, resolution="0.1")
        with self.assertRaises(TypeError):
            build_occupancy_grid(scans, resolution=True)
        with self.assertRaises(TypeError):
            build_occupancy_grid(scans, padding="0")
        with self.assertRaises(TypeError):
            build_occupancy_grid(scans, padding=False)
        with self.assertRaises(TypeError):
            build_occupancy_grid(scans, max_range="1")
        with self.assertRaises(TypeError):
            build_occupancy_grid(scans, max_range=True)


class ValueErrorTest(unittest.TestCase):
    def test_empty_scans_rejected(self):
        with self.assertRaises(ValueError):
            build_occupancy_grid([])
        with self.assertRaises(ValueError):
            build_occupancy_grid(iter(()))

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            build_occupancy_grid(
                [
                    (0, Pose3(), [(1.0, 0.0, 0.0)]),
                    (0, Pose3(translation=(2.0, 0.0, 0.0)), [(1.0, 0.0, 0.0)]),
                ]
            )

    def test_empty_cloud_rejected(self):
        with self.assertRaises(ValueError):
            build_occupancy_grid([(0, Pose3(), [])])
        with self.assertRaises(ValueError):
            build_occupancy_grid([(0, Pose3(), iter(()))])

    def test_non_finite_coordinates_rejected(self):
        with self.assertRaises(ValueError):
            build_occupancy_grid([(0, Pose3(), [(float("nan"), 0.0, 0.0)])])
        with self.assertRaises(ValueError):
            build_occupancy_grid([(0, Pose3(), [(float("inf"), 0.0, 0.0)])])
        with self.assertRaises(ValueError):
            build_occupancy_grid([(0, Pose3(), [(1.0, float("-inf"), 0.0)])])

    def test_resolution_range(self):
        scans = [(0, Pose3(), [(1.0, 0.0, 0.0)])]
        for bad in (0.0, -0.05, float("nan"), float("inf")):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    build_occupancy_grid(scans, resolution=bad)

    def test_padding_range(self):
        scans = [(0, Pose3(), [(1.0, 0.0, 0.0)])]
        for bad in (-0.01, float("nan"), float("inf")):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    build_occupancy_grid(scans, padding=bad)

    def test_max_range_range(self):
        scans = [(0, Pose3(), [(1.0, 0.0, 0.0)])]
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    build_occupancy_grid(scans, max_range=bad)


if __name__ == "__main__":
    unittest.main()
