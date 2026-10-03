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


def scan(frame_id, pose, points):
    return (frame_id, pose, points)


class BasicCastingTest(unittest.TestCase):
    def test_straight_ray_along_x(self):
        grid = build_occupancy_grid(
            [scan(1, Pose3(), [(5.0, 0.0, 0.0)])], resolution=1.0
        )
        self.assertEqual(grid.resolution, 1.0)
        self.assertEqual(grid.origin, (0.0, 0.0))
        self.assertEqual((grid.width, grid.height), (6, 1))
        self.assertEqual(grid.data, (FREE, FREE, FREE, FREE, FREE, OCCUPIED))

    def test_straight_ray_along_y(self):
        grid = build_occupancy_grid(
            [scan(1, Pose3(), [(0.0, 4.0, 0.0)])], resolution=1.0
        )
        self.assertEqual((grid.width, grid.height), (1, 5))
        # data is row-major with y increasing; only the top cell is occupied
        self.assertEqual(grid.data, (FREE, FREE, FREE, FREE, OCCUPIED))

    def test_diagonal_ray_is_bresenham(self):
        # (0,0) -> (3,3) visits the four diagonal cells; endpoint occupied.
        grid = build_occupancy_grid(
            [scan(1, Pose3(), [(3.0, 3.0, 0.0)])], resolution=1.0
        )
        self.assertEqual((grid.width, grid.height), (4, 4))
        expected = [UNKNOWN] * 16
        for k in range(3):
            expected[k * 4 + k] = FREE
        expected[3 * 4 + 3] = OCCUPIED
        self.assertEqual(grid.data, tuple(expected))

    def test_z_is_ignored_in_projection(self):
        grid_a = build_occupancy_grid(
            [scan(1, Pose3(), [(2.0, 0.0, 7.0)])], resolution=1.0
        )
        grid_b = build_occupancy_grid(
            [scan(1, Pose3(), [(2.0, 0.0, -3.0)])], resolution=1.0
        )
        self.assertEqual(grid_a.data, grid_b.data)

    def test_zero_horizontal_distance_occupies_cell(self):
        grid = build_occupancy_grid(
            [scan(1, Pose3(), [(0.0, 0.0, 5.0)])], resolution=1.0
        )
        self.assertEqual((grid.width, grid.height), (1, 1))
        self.assertEqual(grid.data, (OCCUPIED,))

    def test_pose_transforms_points_and_origin(self):
        # Sensor at (1, 2); local x-axis point lands at world (1, 3).
        pose = Pose3(
            translation=(1.0, 2.0, 0.0),
            quaternion=(math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)),
        )
        grid = build_occupancy_grid(
            [scan(7, pose, [(1.0, 0.0, 0.0)])], resolution=1.0
        )
        self.assertEqual(grid.origin, (1.0, 2.0))
        self.assertEqual((grid.width, grid.height), (1, 2))
        self.assertEqual(grid.data, (FREE, OCCUPIED))


class BoundsTest(unittest.TestCase):
    def test_bounds_align_outward_to_resolution(self):
        # Hit at 5.0 with resolution 0.3: tight interval [0, 5] extends to
        # 17 cells (17 * 0.3 = 5.1), origin stays at the sensor.
        grid = build_occupancy_grid(
            [scan(1, Pose3(), [(5.0, 0.0, 0.0)])], resolution=0.3
        )
        self.assertEqual(grid.origin, (0.0, 0.0))
        self.assertEqual(grid.width, 17)
        self.assertEqual(grid.height, 1)
        self.assertEqual(grid.data[-1], OCCUPIED)
        self.assertTrue(all(value == FREE for value in grid.data[:-1]))

    def test_padding_expands_on_all_sides(self):
        grid = build_occupancy_grid(
            [scan(1, Pose3(), [(5.0, 0.0, 0.0)])],
            resolution=1.0,
            padding=1.0,
        )
        # Tight cells 0..5 become outer cells -1..6: 8 columns.
        self.assertEqual(grid.origin, (-1.0, -1.0))
        self.assertEqual((grid.width, grid.height), (8, 3))
        expected = [UNKNOWN] * 24
        # Sensor at cell (1, 1), hit at cell (6, 1).
        for x in range(1, 6):
            expected[1 * 8 + x] = FREE
        expected[1 * 8 + 6] = OCCUPIED
        self.assertEqual(grid.data, tuple(expected))

    def test_bounds_cover_all_sensor_origins(self):
        # Second sensor sits beyond its own ray endpoint; bounds must include
        # its origin cell either way.
        grid = build_occupancy_grid(
            [
                scan(1, Pose3(translation=(0.0, 0.0, 0.0)), [(1.0, 0.0, 0.0)]),
                scan(2, Pose3(translation=(4.0, 0.0, 0.0)), [(-1.0, 0.0, 0.0)]),
            ],
            resolution=1.0,
        )
        # x spans sensor 0 .. sensor 4: five cells.
        self.assertEqual((grid.width, grid.height), (5, 1))
        # Frame 2 ray walks from cell 4 through to cell 3: cell 4 free.
        self.assertEqual(grid.data[4], FREE)


class MergeTest(unittest.TestCase):
    def test_occupied_wins_over_free(self):
        grid = build_occupancy_grid(
            [
                scan(1, Pose3(), [(3.0, 0.0, 0.0)]),
                scan(2, Pose3(translation=(5.0, 0.0, 0.0)), [(-4.0, 0.0, 0.0)]),
            ],
            resolution=1.0,
        )
        # Ray 1: 0->3 (cell 3 occupied, 0..2 free). Ray 2: 5->1 (cell 1
        # occupied; cells 2..5 free). Cell 3 is free for ray 2 but occupied
        # by ray 1; cell 1 is free for ray 1 and occupied by ray 2. Both
        # stay occupied regardless of processing order.
        self.assertEqual(grid.data[1], OCCUPIED)
        self.assertEqual(grid.data[3], OCCUPIED)
        self.assertEqual(grid.data, (FREE, OCCUPIED, FREE, OCCUPIED, FREE, FREE))

    def test_result_independent_of_input_order(self):
        frames = [
            (1, Pose3(translation=(0.0, 0.0, 0.0)), [(2.0, 1.0, 0.0)]),
            (2, Pose3(translation=(3.0, 3.0, 0.0)), [(-1.0, -2.0, 0.0)]),
            (3, Pose3(translation=(1.0, 4.0, 0.0)), [(0.0, -3.0, 2.0)]),
        ]
        forward = build_occupancy_grid(frames, resolution=0.5)
        reversed_ = build_occupancy_grid(list(reversed(frames)), resolution=0.5)
        shuffled = build_occupancy_grid(
            [frames[2], frames[0], frames[1]], resolution=0.5
        )
        self.assertEqual(forward, reversed_)
        self.assertEqual(forward, shuffled)


class MaxRangeTest(unittest.TestCase):
    def test_ray_beyond_range_is_truncated_free_only(self):
        grid = build_occupancy_grid(
            [scan(1, Pose3(), [(10.0, 0.0, 0.0)])],
            resolution=1.0,
            max_range=5.0,
        )
        # Endpoint at the range boundary is free, not occupied.
        self.assertEqual(grid.data, (FREE, FREE, FREE, FREE, FREE, FREE))
        self.assertNotIn(OCCUPIED, grid.data)

    def test_ray_within_range_keeps_hit(self):
        grid = build_occupancy_grid(
            [scan(1, Pose3(), [(4.0, 0.0, 0.0)])],
            resolution=1.0,
            max_range=5.0,
        )
        self.assertEqual(grid.data[-1], OCCUPIED)

    def test_ray_exactly_at_range_occupies(self):
        grid = build_occupancy_grid(
            [scan(1, Pose3(), [(5.0, 0.0, 0.0)])],
            resolution=1.0,
            max_range=5.0,
        )
        self.assertEqual(grid.data[-1], OCCUPIED)

    def test_truncation_uses_horizontal_distance(self):
        # 3D distance 5 but horizontal distance 3: within range 4.
        grid = build_occupancy_grid(
            [scan(1, Pose3(), [(3.0, 0.0, 4.0)])],
            resolution=1.0,
            max_range=4.0,
        )
        self.assertEqual(grid.data[-1], OCCUPIED)


class ImmutabilityAndConsumptionTest(unittest.TestCase):
    def test_return_value_is_immutable(self):
        grid = build_occupancy_grid(
            [scan(1, Pose3(), [(1.0, 0.0, 0.0)])], resolution=1.0
        )
        self.assertIsInstance(grid, OccupancyGrid2D)
        self.assertIsInstance(grid.data, tuple)
        self.assertIsInstance(grid.origin, tuple)

    def test_generators_consumed_once_and_inputs_unchanged(self):
        points = [(2.0, 0.0, 0.0), (1.0, 1.0, 0.0)]

        def cloud_gen():
            yield from points

        def scans_gen():
            yield (1, Pose3(), cloud_gen())

        grid = build_occupancy_grid(scans_gen(), resolution=1.0)
        self.assertEqual(points, [(2.0, 0.0, 0.0), (1.0, 1.0, 0.0)])
        self.assertEqual(len(grid.data), grid.width * grid.height)

    def test_lists_and_tuples_equivalent(self):
        tuple_grid = build_occupancy_grid(
            [(1, Pose3(), [(1.0, 0.0, 0.0)])], resolution=1.0
        )
        list_grid = build_occupancy_grid(
            [[1, Pose3(), [[1.0, 0.0, 0.0]]]], resolution=1.0
        )
        self.assertEqual(tuple_grid, list_grid)


class ValidationTest(unittest.TestCase):
    def test_empty_scans_rejected(self):
        with self.assertRaises(ValueError):
            build_occupancy_grid([])

    def test_empty_cloud_rejected(self):
        with self.assertRaises(ValueError):
            build_occupancy_grid([(1, Pose3(), [])])

    def test_non_iterable_scans_type_error(self):
        with self.assertRaises(TypeError):
            build_occupancy_grid(42)

    def test_malformed_frame_type_error(self):
        with self.assertRaises(TypeError):
            build_occupancy_grid([(1, Pose3())], resolution=1.0)
        with self.assertRaises(TypeError):
            build_occupancy_grid([(1, Pose3(), [], "extra")], resolution=1.0)

    def test_bad_id_type_error(self):
        with self.assertRaises(TypeError):
            build_occupancy_grid([(1.0, Pose3(), [(1.0, 0.0, 0.0)])])
        with self.assertRaises(TypeError):
            build_occupancy_grid([(True, Pose3(), [(1.0, 0.0, 0.0)])])
        with self.assertRaises(TypeError):
            build_occupancy_grid([("1", Pose3(), [(1.0, 0.0, 0.0)])])

    def test_bad_pose_type_error(self):
        with self.assertRaises(TypeError):
            build_occupancy_grid([(1, (0.0, 0.0, 0.0), [(1.0, 0.0, 0.0)])])

    def test_bad_point_shape_type_error(self):
        with self.assertRaises(TypeError):
            build_occupancy_grid([(1, Pose3(), [(1.0, 0.0)])])
        with self.assertRaises(TypeError):
            build_occupancy_grid([(1, Pose3(), [(1.0, 0.0, 0.0, 0.0)])])

    def test_non_real_coordinate_type_error(self):
        with self.assertRaises(TypeError):
            build_occupancy_grid([(1, Pose3(), [("1", 0.0, 0.0)])])
        with self.assertRaises(TypeError):
            build_occupancy_grid([(1, Pose3(), [(True, 0.0, 0.0)])])

    def test_non_finite_coordinate_value_error(self):
        with self.assertRaises(ValueError):
            build_occupancy_grid([(1, Pose3(), [(math.nan, 0.0, 0.0)])])
        with self.assertRaises(ValueError):
            build_occupancy_grid([(1, Pose3(), [(math.inf, 0.0, 0.0)])])

    def test_duplicate_id_value_error(self):
        with self.assertRaises(ValueError):
            build_occupancy_grid(
                [
                    (1, Pose3(), [(1.0, 0.0, 0.0)]),
                    (1, Pose3(translation=(2.0, 0.0, 0.0)), [(1.0, 0.0, 0.0)]),
                ],
                resolution=1.0,
            )

    def test_bad_resolution(self):
        scans = [(1, Pose3(), [(1.0, 0.0, 0.0)])]
        for bad in (0.0, -1.0, math.nan, math.inf):
            with self.assertRaises(ValueError):
                build_occupancy_grid(scans, resolution=bad)
        with self.assertRaises(TypeError):
            build_occupancy_grid(scans, resolution="0.1")
        with self.assertRaises(TypeError):
            build_occupancy_grid(scans, resolution=True)

    def test_bad_padding(self):
        scans = [(1, Pose3(), [(1.0, 0.0, 0.0)])]
        for bad in (-0.1, math.nan, math.inf):
            with self.assertRaises(ValueError):
                build_occupancy_grid(scans, resolution=1.0, padding=bad)
        with self.assertRaises(TypeError):
            build_occupancy_grid(scans, resolution=1.0, padding="1")

    def test_bad_max_range(self):
        scans = [(1, Pose3(), [(1.0, 0.0, 0.0)])]
        for bad in (0.0, -1.0, math.nan, math.inf):
            with self.assertRaises(ValueError):
                build_occupancy_grid(scans, resolution=1.0, max_range=bad)
        with self.assertRaises(TypeError):
            build_occupancy_grid(scans, resolution=1.0, max_range="5")
        with self.assertRaises(TypeError):
            build_occupancy_grid(scans, resolution=1.0, max_range=True)

    def test_defaults_match_explicit(self):
        scans = [(1, Pose3(), [(0.5, 0.25, 0.0)])]
        self.assertEqual(
            build_occupancy_grid(scans),
            build_occupancy_grid(scans, resolution=0.05, padding=0.0, max_range=None),
        )


if __name__ == "__main__":
    unittest.main()
