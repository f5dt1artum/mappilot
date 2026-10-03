import math
import unittest

from mappilot.geometry import Pose3
from mappilot.localization import CorrelativeScanMatchResult, correlative_scan_match
from mappilot.mapping import OccupancyGrid2D, build_occupancy_grid


def yaw_pose(x, y, z, yaw):
    return Pose3((x, y, z), (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)))


def assert_pose_almost_equal(test_case, actual, expected):
    for actual_value, expected_value in zip(actual.translation, expected.translation):
        test_case.assertAlmostEqual(actual_value, expected_value, places=12)
    for actual_value, expected_value in zip(actual.quaternion, expected.quaternion):
        test_case.assertAlmostEqual(actual_value, expected_value, places=12)


SCAN = (
    (1.0, 0.0, 0.0),
    (0.0, 2.0, 0.0),
    (-1.5, 0.5, 0.0),
    (0.5, -1.0, 0.0),
    (2.0, 2.0, 0.0),
)


def recovery_grid():
    true_pose = yaw_pose(2.0, 3.0, 0.5, 0.3)
    grid = build_occupancy_grid([(0, true_pose, SCAN)], resolution=0.05)
    return grid, true_pose


class RecoveryTest(unittest.TestCase):
    def test_recovers_true_pose_within_windows(self):
        grid, true_pose = recovery_grid()
        initial = yaw_pose(2.1, 2.9, 0.5, 0.2)
        result = correlative_scan_match(
            grid, SCAN, initial,
            x_window=0.12, x_step=0.05,
            y_window=0.12, y_step=0.05,
            yaw_window=0.12, yaw_step=0.05,
            min_known_points=3, min_score=0.5,
        )
        self.assertIsInstance(result, CorrelativeScanMatchResult)
        self.assertTrue(result.matched)
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.known_points, len(SCAN))
        self.assertEqual(result.evaluated_candidates, 5 * 5 * 5)
        assert_pose_almost_equal(self, result.pose, true_pose)

    def test_z_is_preserved(self):
        grid, _true_pose = recovery_grid()
        initial = yaw_pose(2.1, 2.9, 0.7, 0.2)
        result = correlative_scan_match(
            grid, SCAN, initial,
            x_window=0.12, x_step=0.05,
            y_window=0.12, y_step=0.05,
            yaw_window=0.12, yaw_step=0.05,
            min_known_points=3, min_score=0.5,
        )
        self.assertTrue(result.matched)
        self.assertEqual(result.pose.translation[2], 0.7)

    def test_zero_windows_evaluate_only_initial_pose(self):
        grid, _true_pose = recovery_grid()
        initial = yaw_pose(2.0, 3.0, 0.5, 0.3)
        result = correlative_scan_match(
            grid, SCAN, initial,
            x_window=0.0, x_step=0.05,
            y_window=0.0, y_step=0.05,
            yaw_window=0.0, yaw_step=0.05,
            min_known_points=1, min_score=0.5,
        )
        self.assertEqual(result.evaluated_candidates, 1)
        self.assertTrue(result.matched)
        self.assertEqual(result.pose, initial)

    def test_result_is_deterministic(self):
        grid, _true_pose = recovery_grid()
        initial = yaw_pose(2.1, 2.9, 0.5, 0.2)
        arguments = dict(
            x_window=0.12, x_step=0.05,
            y_window=0.12, y_step=0.05,
            yaw_window=0.12, yaw_step=0.05,
            min_known_points=3, min_score=0.5,
        )
        first = correlative_scan_match(grid, SCAN, initial, **arguments)
        second = correlative_scan_match(grid, SCAN, initial, **arguments)
        self.assertEqual(first, second)

    def test_generator_scan_is_consumed_once(self):
        grid, true_pose = recovery_grid()
        initial = yaw_pose(2.1, 2.9, 0.5, 0.2)
        result = correlative_scan_match(
            grid, (point for point in SCAN), initial,
            x_window=0.12, x_step=0.05,
            y_window=0.12, y_step=0.05,
            yaw_window=0.12, yaw_step=0.05,
            min_known_points=3, min_score=0.5,
        )
        assert_pose_almost_equal(self, result.pose, true_pose)

    def test_inputs_are_not_modified(self):
        grid, _true_pose = recovery_grid()
        scan = [list(point) for point in SCAN]
        snapshot = [list(point) for point in SCAN]
        initial = yaw_pose(2.1, 2.9, 0.5, 0.2)
        correlative_scan_match(
            grid, scan, initial,
            x_window=0.12, x_step=0.05,
            y_window=0.12, y_step=0.05,
            yaw_window=0.12, yaw_step=0.05,
            min_known_points=3, min_score=0.5,
        )
        self.assertEqual(scan, snapshot)


class ScoringTest(unittest.TestCase):
    def simple_grid(self):
        # Cell (0, 0) occupied, cell (1, 0) free.
        return OccupancyGrid2D(
            resolution=1.0, origin=(0.0, 0.0), width=2, height=1, data=(100, 0)
        )

    def test_below_threshold_returns_best_pose_unmatched(self):
        initial = Pose3.identity()
        result = correlative_scan_match(
            self.simple_grid(),
            [(0.5, 0.5, 0.0), (1.5, 0.5, 0.0)],
            initial,
            x_window=0.0, x_step=1.0,
            y_window=0.0, y_step=1.0,
            yaw_window=0.0, yaw_step=1.0,
            min_known_points=1, min_score=0.5,
        )
        self.assertFalse(result.matched)
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.known_points, 2)
        self.assertEqual(result.evaluated_candidates, 1)
        self.assertEqual(result.pose, initial)

    def test_score_at_threshold_matches(self):
        result = correlative_scan_match(
            self.simple_grid(),
            [(0.5, 0.5, 0.0), (1.5, 0.5, 0.0)],
            Pose3.identity(),
            x_window=0.0, x_step=1.0,
            y_window=0.0, y_step=1.0,
            yaw_window=0.0, yaw_step=1.0,
            min_known_points=1, min_score=0.0,
        )
        self.assertTrue(result.matched)
        self.assertEqual(result.score, 0.0)

    def test_unknown_and_out_of_map_points_are_not_known(self):
        grid = OccupancyGrid2D(
            resolution=1.0, origin=(0.0, 0.0), width=2, height=1, data=(100, -1)
        )
        result = correlative_scan_match(
            grid,
            [(0.5, 0.5, 0.0), (1.5, 0.5, 0.0), (50.0, 50.0, 0.0)],
            Pose3.identity(),
            x_window=0.0, x_step=1.0,
            y_window=0.0, y_step=1.0,
            yaw_window=0.0, yaw_step=1.0,
            min_known_points=1, min_score=-1.0,
        )
        self.assertEqual(result.known_points, 1)
        self.assertEqual(result.score, 1.0)

    def test_no_qualified_candidate_returns_initial_pose(self):
        initial = Pose3.identity()
        result = correlative_scan_match(
            self.simple_grid(),
            [(0.5, 0.5, 0.0), (1.5, 0.5, 0.0)],
            initial,
            x_window=0.0, x_step=1.0,
            y_window=0.0, y_step=1.0,
            yaw_window=0.0, yaw_step=1.0,
            min_known_points=3, min_score=0.5,
        )
        self.assertFalse(result.matched)
        self.assertIsNone(result.score)
        self.assertEqual(result.known_points, 0)
        self.assertEqual(result.evaluated_candidates, 1)
        self.assertIs(result.pose, initial)

    def test_tie_breaks_to_smaller_offset_then_ascending(self):
        # Occupied cells at ix 0 and 2, free cell at ix 1; the scan point sits
        # over the free cell, so dx=-1 and dx=+1 tie on score, known count and
        # squared offset; ascending dx wins.
        grid = OccupancyGrid2D(
            resolution=1.0, origin=(0.0, 0.0), width=3, height=1,
            data=(100, 0, 100),
        )
        result = correlative_scan_match(
            grid,
            [(1.5, 0.5, 0.0)],
            Pose3.identity(),
            x_window=1.0, x_step=1.0,
            y_window=0.0, y_step=1.0,
            yaw_window=0.0, yaw_step=1.0,
            min_known_points=1, min_score=0.5,
        )
        self.assertTrue(result.matched)
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.pose.translation, (-1.0, 0.0, 0.0))

    def test_more_known_points_wins_tie(self):
        # Both candidates score 1.0; the zero offset keeps both points on
        # occupied cells while dx=-1 pushes one point out of the map.
        grid = OccupancyGrid2D(
            resolution=1.0, origin=(0.0, 0.0), width=2, height=1,
            data=(100, 100),
        )
        result = correlative_scan_match(
            grid,
            [(0.5, 0.5, 0.0), (1.5, 0.5, 0.0)],
            Pose3.identity(),
            x_window=1.0, x_step=1.0,
            y_window=0.0, y_step=1.0,
            yaw_window=0.0, yaw_step=1.0,
            min_known_points=1, min_score=0.5,
        )
        self.assertEqual(result.known_points, 2)
        self.assertEqual(result.pose.translation, (0.0, 0.0, 0.0))


class ValidationTest(unittest.TestCase):
    def setUp(self):
        self.grid, _true_pose = recovery_grid()
        self.initial = yaw_pose(2.1, 2.9, 0.5, 0.2)
        self.arguments = dict(
            x_window=0.12, x_step=0.05,
            y_window=0.12, y_step=0.05,
            yaw_window=0.12, yaw_step=0.05,
            min_known_points=3, min_score=0.5,
        )

    def match(self, **overrides):
        arguments = dict(self.arguments)
        arguments.update(overrides)
        grid = arguments.pop("grid", self.grid)
        scan = arguments.pop("scan", SCAN)
        initial = arguments.pop("initial_pose", self.initial)
        return correlative_scan_match(grid, scan, initial, **arguments)

    def test_type_errors(self):
        bad_calls = [
            dict(grid=[[100]]),
            dict(initial_pose=(0.0, 0.0, 0.0)),
            dict(scan=42),
            dict(scan="not a cloud"),
            dict(scan=[(1.0, 2.0)]),
            dict(scan=[(1.0, 2.0, True)]),
            dict(scan=[(1.0, 2.0, "0")]),
            dict(x_window=True),
            dict(x_window="0.1"),
            dict(x_step=None),
            dict(y_window=False),
            dict(y_step="0.05"),
            dict(yaw_window=object()),
            dict(yaw_step=None),
            dict(min_known_points=True),
            dict(min_known_points=1.5),
            dict(min_known_points="3"),
            dict(min_score=True),
            dict(min_score="0.5"),
        ]
        for overrides in bad_calls:
            with self.subTest(overrides=overrides):
                with self.assertRaises(TypeError):
                    self.match(**overrides)

    def test_value_errors(self):
        bad_calls = [
            dict(scan=[]),
            dict(scan=[(1.0, 2.0, math.nan)]),
            dict(scan=[(1.0, 2.0, math.inf)]),
            dict(x_window=-0.1),
            dict(x_window=math.inf),
            dict(x_window=math.nan),
            dict(x_step=0.0),
            dict(x_step=-0.05),
            dict(x_step=math.inf),
            dict(y_window=-1.0),
            dict(y_window=math.inf),
            dict(y_step=0.0),
            dict(y_step=math.nan),
            dict(yaw_window=-0.5),
            dict(yaw_window=math.inf),
            dict(yaw_step=-0.1),
            dict(yaw_step=math.inf),
            dict(min_known_points=0),
            dict(min_known_points=-2),
            dict(min_score=1.5),
            dict(min_score=-1.5),
            dict(min_score=math.inf),
            dict(min_score=math.nan),
        ]
        for overrides in bad_calls:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    self.match(**overrides)

    def test_boundary_min_score_is_accepted(self):
        for threshold in (-1.0, 1.0):
            with self.subTest(threshold=threshold):
                result = self.match(min_score=threshold)
                self.assertIsNotNone(result.score)


if __name__ == "__main__":
    unittest.main()
