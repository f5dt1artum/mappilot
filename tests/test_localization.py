import math
import unittest

from mappilot.geometry import Pose3
from mappilot.localization import (
    CorrelativeScanMatchResult,
    correlative_scan_match,
)
from mappilot.mapping import FREE, OCCUPIED, UNKNOWN, OccupancyGrid2D


def make_grid(data, width, height, resolution=1.0, origin=(0.0, 0.0)):
    return OccupancyGrid2D(
        resolution=resolution,
        origin=origin,
        width=width,
        height=height,
        data=tuple(data),
    )


class TranslationSearchTest(unittest.TestCase):
    def setUp(self):
        # Occupied cells at ix=1 and ix=3 in a single row.
        self.grid = make_grid([FREE, OCCUPIED, FREE, OCCUPIED, FREE], 5, 1)
        self.scan = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
        self.initial = Pose3((0.25, 0.5, 7.0))

    def test_recovers_translation_and_preserves_z(self):
        result = correlative_scan_match(
            self.grid,
            self.scan,
            self.initial,
            x_window=1.0,
            x_step=0.25,
            min_score=0.9,
        )
        # dx=0.75 lands the points on cells 1 and 3 (score 1.0, 2 known);
        # dx=1.0 also scores 1.0 but loses on the smaller norm, and dx=-0.75
        # scores 1.0 with only 1 known point.
        self.assertTrue(result.matched)
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.known_points, 2)
        self.assertEqual(result.evaluated_candidates, 9)
        self.assertEqual(result.pose.translation, (1.0, 0.5, 7.0))
        self.assertEqual(result.pose.quaternion, self.initial.quaternion)

    def test_zero_window_searches_only_initial_pose(self):
        result = correlative_scan_match(self.grid, self.scan, self.initial)
        self.assertEqual(result.evaluated_candidates, 1)
        self.assertEqual(result.pose, self.initial)
        # Both points land on free cells at the initial pose.
        self.assertEqual(result.score, -1.0)
        self.assertEqual(result.known_points, 2)
        self.assertFalse(result.matched)

    def test_evaluated_candidates_counts_full_grid(self):
        result = correlative_scan_match(
            self.grid,
            self.scan,
            self.initial,
            x_window=0.4,
            x_step=0.2,
            y_window=0.4,
            y_step=0.2,
            yaw_window=0.3,
            yaw_step=0.15,
        )
        self.assertEqual(result.evaluated_candidates, 5 * 5 * 5)

    def test_window_excludes_offsets_beyond_it(self):
        # 0.3/0.2 floors to one step: offsets are exactly {-0.2, 0.0, 0.2}.
        result = correlative_scan_match(
            self.grid, self.scan, self.initial, x_window=0.3, x_step=0.2
        )
        self.assertEqual(result.evaluated_candidates, 3)


class YawSearchTest(unittest.TestCase):
    def test_yaw_rotation_about_world_z(self):
        # One occupied cell at ix=0; the scan point must rotate by pi to hit it.
        grid = make_grid([OCCUPIED, FREE], 2, 1)
        scan = [(0.5, 0.0, 0.0)]
        initial = Pose3((1.0, 0.5, 3.0))
        result = correlative_scan_match(
            grid,
            scan,
            initial,
            yaw_window=math.pi,
            yaw_step=math.pi,
            min_score=0.5,
        )
        self.assertTrue(result.matched)
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.evaluated_candidates, 3)
        # dyaw=+pi and dyaw=-pi tie on score, known points and norm; the
        # ascending-dyaw tie-break picks -pi. Both rotate the point onto the
        # occupied cell while z stays put.
        moved = result.pose.transform_point(scan[0])
        self.assertAlmostEqual(moved[0], 0.5)
        self.assertAlmostEqual(moved[1], 0.5)
        self.assertAlmostEqual(moved[2], 3.0)
        self.assertEqual(result.pose.translation, (1.0, 0.5, 3.0))

    def test_dyaw_composes_on_the_left_of_initial_rotation(self):
        # Initial pose already yawed by pi/2; a further dyaw of pi/2 must
        # rotate the scan point by pi in total about the world z axis.
        grid = make_grid([OCCUPIED, FREE], 2, 1)
        scan = [(0.5, 0.0, 0.0)]
        initial = Pose3((1.0, 0.5, 0.0), (math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)))
        result = correlative_scan_match(
            grid,
            scan,
            initial,
            yaw_window=math.pi / 2,
            yaw_step=math.pi / 2,
            min_score=0.5,
        )
        self.assertTrue(result.matched)
        moved = result.pose.transform_point(scan[0])
        self.assertAlmostEqual(moved[0], 0.5)
        self.assertAlmostEqual(moved[1], 0.5)


class ScoringTest(unittest.TestCase):
    def test_unknown_and_out_of_map_points_are_not_counted(self):
        grid = make_grid([OCCUPIED, UNKNOWN], 2, 1)
        scan = [(0.5, 0.5, 0.0), (1.5, 0.5, 0.0), (5.0, 0.5, 0.0)]
        result = correlative_scan_match(grid, scan, Pose3())
        # Only the occupied hit counts: unknown and out-of-map points are
        # neither known nor scored.
        self.assertEqual(result.known_points, 1)
        self.assertEqual(result.score, 1.0)

    def test_free_cells_score_minus_one(self):
        grid = make_grid([FREE, OCCUPIED], 2, 1)
        scan = [(0.5, 0.5, 0.0), (1.5, 0.5, 0.0)]
        result = correlative_scan_match(grid, scan, Pose3())
        self.assertEqual(result.known_points, 2)
        self.assertEqual(result.score, 0.0)

    def test_best_below_threshold_still_returned_unmatched(self):
        grid = make_grid([FREE, OCCUPIED], 2, 1)
        scan = [(0.5, 0.5, 0.0), (1.5, 0.5, 0.0)]
        initial = Pose3()
        result = correlative_scan_match(grid, scan, initial, min_score=0.5)
        self.assertFalse(result.matched)
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.pose, initial)

    def test_no_candidate_reaching_min_known_points(self):
        grid = make_grid([OCCUPIED, FREE], 2, 1)
        scan = [(0.5, 0.5, 0.0)]
        initial = Pose3((0.0, 0.0, 1.0))
        result = correlative_scan_match(
            grid,
            scan,
            initial,
            x_window=0.5,
            x_step=0.25,
            min_known_points=2,
        )
        self.assertFalse(result.matched)
        self.assertIsNone(result.score)
        self.assertEqual(result.known_points, 0)
        self.assertIs(result.pose, initial)
        self.assertEqual(result.evaluated_candidates, 5)

    def test_result_is_immutable_named_tuple(self):
        grid = make_grid([OCCUPIED], 1, 1)
        result = correlative_scan_match(grid, [(0.5, 0.5, 0.0)], Pose3())
        self.assertIsInstance(result, CorrelativeScanMatchResult)
        with self.assertRaises(AttributeError):
            result.score = 1.0


class TieBreakTest(unittest.TestCase):
    def test_zero_offset_wins_ties(self):
        # Symmetric occupied cells: dx=+1 and dx=-1 both score 1.0 with two
        # known points, but dx=0 also scores 1.0 and has the smallest norm.
        grid = make_grid([OCCUPIED, FREE, OCCUPIED], 3, 1)
        scan = [(0.5, 0.5, 0.0), (2.5, 0.5, 0.0)]
        result = correlative_scan_match(
            grid, scan, Pose3(), x_window=1.0, x_step=1.0
        )
        self.assertEqual(result.pose.translation, (0.0, 0.0, 0.0))
        self.assertEqual(result.score, 1.0)

    def test_more_known_points_wins_score_ties(self):
        # dx=0 scores 1.0 with one known point (the other is out of map);
        # dx=0.5 scores 1.0 with two known points and wins despite the
        # larger norm.
        grid = make_grid([OCCUPIED, OCCUPIED], 2, 1)
        scan = [(-0.5, 0.5, 0.0), (0.5, 0.5, 9.0)]
        result = correlative_scan_match(
            grid, scan, Pose3(), x_window=0.5, x_step=0.5
        )
        self.assertEqual(result.known_points, 2)
        self.assertEqual(result.pose.translation, (0.5, 0.0, 0.0))


class RobustnessTest(unittest.TestCase):
    def setUp(self):
        self.grid = make_grid([OCCUPIED, FREE], 2, 1)
        self.scan = [(0.5, 0.5, 0.0)]

    def test_generator_consumed_once(self):
        result = correlative_scan_match(
            self.grid, (point for point in self.scan), Pose3()
        )
        self.assertEqual(result.score, 1.0)

    def test_input_scan_not_modified(self):
        scan = [[0.5, 0.5, 0.0]]
        correlative_scan_match(self.grid, scan, Pose3())
        self.assertEqual(scan, [[0.5, 0.5, 0.0]])

    def test_deterministic_for_same_input(self):
        kwargs = dict(x_window=0.5, x_step=0.25, yaw_window=0.4, yaw_step=0.2)
        first = correlative_scan_match(self.grid, self.scan, Pose3(), **kwargs)
        second = correlative_scan_match(self.grid, self.scan, Pose3(), **kwargs)
        self.assertEqual(first, second)

    def test_grid_from_builder_matches(self):
        from mappilot.mapping import build_occupancy_grid

        built = build_occupancy_grid(
            [(0, Pose3(), [(2.0, 0.0, 0.0)])], resolution=0.5
        )
        result = correlative_scan_match(
            built, [(2.0, 0.0, 0.0)], Pose3(), min_score=0.5
        )
        self.assertTrue(result.matched)
        self.assertEqual(result.score, 1.0)


class ValidationTest(unittest.TestCase):
    def setUp(self):
        self.grid = make_grid([OCCUPIED, FREE], 2, 1)
        self.scan = [(0.5, 0.5, 0.0)]
        self.pose = Pose3()

    def test_type_errors(self):
        cases = [
            dict(grid="not a grid"),
            dict(scan_points=5),
            dict(scan_points=["abc"]),
            dict(scan_points=[(0.0, 0.0)]),
            dict(scan_points=[(0.0, 0.0, "x")]),
            dict(scan_points=[(0.0, 0.0, True)]),
            dict(initial_pose="not a pose"),
            dict(x_window="wide"),
            dict(x_window=True),
            dict(y_step=None),
            dict(yaw_window=object()),
            dict(min_known_points=True),
            dict(min_known_points=1.5),
            dict(min_score="high"),
        ]
        for overrides in cases:
            with self.subTest(**overrides):
                kwargs = dict(
                    grid=self.grid,
                    scan_points=self.scan,
                    initial_pose=self.pose,
                )
                kwargs.update(overrides)
                grid = kwargs.pop("grid")
                scan = kwargs.pop("scan_points")
                pose = kwargs.pop("initial_pose")
                with self.assertRaises(TypeError):
                    correlative_scan_match(grid, scan, pose, **kwargs)

    def test_value_errors(self):
        cases = [
            dict(scan_points=[]),
            dict(scan_points=[(0.0, 0.0, float("nan"))]),
            dict(scan_points=[(0.0, float("inf"), 0.0)]),
            dict(x_window=-0.1),
            dict(x_window=float("inf")),
            dict(y_window=float("nan")),
            dict(y_step=0.0),
            dict(y_step=-1.0),
            dict(yaw_step=float("inf")),
            dict(yaw_window=-0.5),
            dict(min_known_points=0),
            dict(min_known_points=-3),
            dict(min_score=1.5),
            dict(min_score=-1.5),
            dict(min_score=float("nan")),
        ]
        for overrides in cases:
            with self.subTest(**{k: repr(v) for k, v in overrides.items()}):
                kwargs = dict(
                    grid=self.grid,
                    scan_points=self.scan,
                    initial_pose=self.pose,
                )
                kwargs.update(overrides)
                grid = kwargs.pop("grid")
                scan = kwargs.pop("scan_points")
                pose = kwargs.pop("initial_pose")
                with self.assertRaises(ValueError):
                    correlative_scan_match(grid, scan, pose, **kwargs)

    def test_boundary_min_score_accepted(self):
        for threshold in (-1.0, 1.0):
            result = correlative_scan_match(
                self.grid, self.scan, self.pose, min_score=threshold
            )
            self.assertIsNotNone(result.score)


if __name__ == "__main__":
    unittest.main()
