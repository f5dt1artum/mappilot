import math
import unittest

from mappilot.geometry import Pose3
from mappilot.motion_compensation import deskew_point_cloud

SQRT2_HALF = math.sqrt(0.5)


def static_trajectory(times=(0.0, 1.0)):
    return [(t, Pose3.identity()) for t in times]


class DeskewStaticTest(unittest.TestCase):
    def test_static_trajectory_preserves_points_exactly(self):
        points = [(0.0, (1.0, 2.0, 3.0)), (0.5, (-4.0, 5.0, -6.0)), (1.0, (0.0, 0.0, 0.0))]
        result = deskew_point_cloud(points, 10.0, static_trajectory((10.0, 11.0)), Pose3.identity())
        self.assertEqual(result, tuple(point for _, point in points))

    def test_static_trajectory_with_extrinsic_preserves_points(self):
        extrinsic = Pose3((1.0, 2.0, 3.0), (SQRT2_HALF, 0.0, 0.0, SQRT2_HALF))
        points = [(0.25, (1.0, 0.0, 0.0)), (0.75, (0.0, 1.0, 2.0))]
        result = deskew_point_cloud(points, 0.0, static_trajectory(), extrinsic)
        for expected, actual in zip((p for _, p in points), result):
            self.assertAlmostEqual(expected[0], actual[0])
            self.assertAlmostEqual(expected[1], actual[1])
            self.assertAlmostEqual(expected[2], actual[2])

    def test_result_is_immutable_tuples_in_input_order(self):
        points = [(0.9, (3.0, 0.0, 0.0)), (0.1, (1.0, 0.0, 0.0)), (0.5, (2.0, 0.0, 0.0))]
        result = deskew_point_cloud(points, 0.0, static_trajectory(), Pose3.identity())
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)
        for actual, expected in zip(result, ((3.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0))):
            self.assertIsInstance(actual, tuple)
            self.assertEqual(actual, expected)


class DeskewMotionTest(unittest.TestCase):
    def test_uniform_translation(self):
        trajectory = [(0.0, Pose3.identity()), (1.0, Pose3((1.0, 0.0, 0.0)))]
        points = [(0.5, (0.0, 0.0, 0.0)), (0.0, (2.0, 1.0, 3.0))]
        result = deskew_point_cloud(points, 0.0, trajectory, Pose3.identity())
        self.assertEqual(result[0], (0.5, 0.0, 0.0))
        self.assertEqual(result[1], (2.0, 1.0, 3.0))

    def test_uniform_rotation_slerp_halfway(self):
        trajectory = [
            (0.0, Pose3.identity()),
            (1.0, Pose3((0.0, 0.0, 0.0), (SQRT2_HALF, 0.0, 0.0, SQRT2_HALF))),
        ]
        result = deskew_point_cloud([(0.5, (1.0, 0.0, 0.0))], 0.0, trajectory, Pose3.identity())
        self.assertAlmostEqual(result[0][0], SQRT2_HALF)
        self.assertAlmostEqual(result[0][1], SQRT2_HALF)
        self.assertAlmostEqual(result[0][2], 0.0)

    def test_exact_sample_hit_uses_sample_pose(self):
        trajectory = [
            (0.0, Pose3.identity()),
            (0.5, Pose3((10.0, 0.0, 0.0))),
            (1.0, Pose3((10.0, 0.0, 0.0))),
        ]
        result = deskew_point_cloud([(0.5, (1.0, 2.0, 3.0))], 0.0, trajectory, Pose3.identity())
        self.assertEqual(result[0], (11.0, 2.0, 3.0))

    def test_reference_time_moves_output_frame(self):
        trajectory = [(0.0, Pose3.identity()), (1.0, Pose3((1.0, 0.0, 0.0)))]
        points = [(0.0, (0.0, 0.0, 0.0)), (1.0, (0.0, 0.0, 0.0))]
        result = deskew_point_cloud(
            points, 0.0, trajectory, Pose3.identity(), reference_time=1.0
        )
        self.assertEqual(result[0], (-1.0, 0.0, 0.0))
        self.assertEqual(result[1], (0.0, 0.0, 0.0))

    def test_extrinsic_is_applied_at_sample_and_reference_times(self):
        extrinsic = Pose3((0.0, 1.0, 0.0))
        trajectory = [(0.0, Pose3.identity()), (1.0, Pose3((2.0, 0.0, 0.0)))]
        result = deskew_point_cloud([(1.0, (0.0, 0.0, 0.0))], 0.0, trajectory, extrinsic)
        self.assertEqual(result[0], (2.0, 0.0, 0.0))

    def test_generator_inputs_are_consumed_once(self):
        points = ((offset, (float(index), 0.0, 0.0)) for index, offset in enumerate((0.0, 0.5, 1.0)))
        trajectory = (record for record in static_trajectory())
        result = deskew_point_cloud(points, 0.0, trajectory, Pose3.identity())
        self.assertEqual(result, ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)))

    def test_caller_objects_are_not_modified(self):
        point_records = [[0.0, [1.0, 2.0, 3.0]], [1.0, [4.0, 5.0, 6.0]]]
        trajectory_records = [[0.0, Pose3.identity()], [1.0, Pose3((1.0, 0.0, 0.0))]]
        deskew_point_cloud(point_records, 0.0, trajectory_records, Pose3.identity())
        self.assertEqual(point_records, [[0.0, [1.0, 2.0, 3.0]], [1.0, [4.0, 5.0, 6.0]]])
        self.assertEqual(len(trajectory_records), 2)


class DeskewGapTest(unittest.TestCase):
    def test_gap_above_limit_raises(self):
        trajectory = [(0.0, Pose3.identity()), (2.0, Pose3.identity())]
        with self.assertRaises(ValueError):
            deskew_point_cloud(
                [(1.0, (0.0, 0.0, 0.0))], 0.0, trajectory, Pose3.identity(),
                max_interpolation_gap=1.0,
            )

    def test_exact_hit_is_exempt_from_gap_limit(self):
        trajectory = [(0.0, Pose3.identity()), (2.0, Pose3((1.0, 0.0, 0.0)))]
        result = deskew_point_cloud(
            [(2.0, (1.0, 0.0, 0.0))], 0.0, trajectory, Pose3.identity(),
            max_interpolation_gap=0.5,
        )
        self.assertEqual(result[0], (2.0, 0.0, 0.0))

    def test_gap_equal_to_limit_is_allowed(self):
        trajectory = [(0.0, Pose3.identity()), (1.0, Pose3.identity())]
        result = deskew_point_cloud(
            [(0.5, (1.0, 2.0, 3.0))], 0.0, trajectory, Pose3.identity(),
            max_interpolation_gap=1.0,
        )
        self.assertEqual(result[0], (1.0, 2.0, 3.0))

    def test_reference_time_interpolation_respects_gap(self):
        trajectory = [(0.0, Pose3.identity()), (2.0, Pose3.identity())]
        with self.assertRaises(ValueError):
            deskew_point_cloud(
                [(0.0, (0.0, 0.0, 0.0))], 0.0, trajectory, Pose3.identity(),
                reference_time=1.0, max_interpolation_gap=0.5,
            )

    def test_invalid_max_interpolation_gap(self):
        args = ([(0.0, (0.0, 0.0, 0.0))], 0.0, static_trajectory(), Pose3.identity())
        for bad in (0.0, -1.0, math.inf, math.nan):
            with self.assertRaises(ValueError):
                deskew_point_cloud(*args, max_interpolation_gap=bad)
        for bad in ("1", True):
            with self.assertRaises(TypeError):
                deskew_point_cloud(*args, max_interpolation_gap=bad)


class DeskewValidationTest(unittest.TestCase):
    def test_type_errors(self):
        good_points = [(0.0, (0.0, 0.0, 0.0))]
        good_trajectory = static_trajectory()
        bad_calls = [
            ("not-iterable", 0.0, good_trajectory, Pose3.identity()),
            ([(0.0, (0.0, 0.0, 0.0))], "0.0", good_trajectory, Pose3.identity()),
            ([(0.0, (0.0, 0.0, 0.0))], True, good_trajectory, Pose3.identity()),
            ([((0.0, 0.0, 0.0),)], 0.0, good_trajectory, Pose3.identity()),
            ([(0.0, 0.0, 0.0)], 0.0, good_trajectory, Pose3.identity()),
            ([(0.0, (0.0, 0.0))], 0.0, good_trajectory, Pose3.identity()),
            ([(0.0, (0.0, 0.0, 0.0, 0.0))], 0.0, good_trajectory, Pose3.identity()),
            ([(True, (0.0, 0.0, 0.0))], 0.0, good_trajectory, Pose3.identity()),
            ([(0.0, (0.0, "x", 0.0))], 0.0, good_trajectory, Pose3.identity()),
            (good_points, 0.0, "not-iterable", Pose3.identity()),
            (good_points, 0.0, [(0.0, Pose3.identity()), (1.0, "pose")], Pose3.identity()),
            (good_points, 0.0, [(0.0, Pose3.identity()), ("t", Pose3.identity())], Pose3.identity()),
            (good_points, 0.0, good_trajectory, ((1.0, 0.0, 0.0))),
            (good_points, 0.0, good_trajectory, Pose3.identity(), "ref"),
        ]
        for args in bad_calls:
            with self.assertRaises(TypeError, msg=repr(args)):
                deskew_point_cloud(*args)

    def test_value_errors(self):
        good_points = [(0.0, (0.0, 0.0, 0.0))]
        good_trajectory = static_trajectory()
        bad_calls = [
            ([], 0.0, good_trajectory, Pose3.identity()),
            (good_points, 0.0, [], Pose3.identity()),
            (good_points, 0.0, [(0.0, Pose3.identity())], Pose3.identity()),
            (good_points, 0.0, [(1.0, Pose3.identity()), (1.0, Pose3.identity())], Pose3.identity()),
            (good_points, 0.0, [(1.0, Pose3.identity()), (0.0, Pose3.identity())], Pose3.identity()),
            ([(math.nan, (0.0, 0.0, 0.0))], 0.0, good_trajectory, Pose3.identity()),
            ([(0.0, (math.inf, 0.0, 0.0))], 0.0, good_trajectory, Pose3.identity()),
            (good_points, math.nan, good_trajectory, Pose3.identity()),
            ([(2.0, (0.0, 0.0, 0.0))], 0.0, good_trajectory, Pose3.identity()),
            ([(-0.1, (0.0, 0.0, 0.0))], 0.0, good_trajectory, Pose3.identity()),
            (good_points, 0.0, good_trajectory, Pose3.identity(), 2.0),
            (good_points, 0.0, good_trajectory, Pose3.identity(), math.inf),
        ]
        for args in bad_calls:
            with self.assertRaises(ValueError, msg=repr(args)):
                deskew_point_cloud(*args)

    def test_boundary_times_are_accepted(self):
        points = [(0.0, (1.0, 0.0, 0.0)), (1.0, (2.0, 0.0, 0.0))]
        result = deskew_point_cloud(points, 5.0, static_trajectory((5.0, 6.0)), Pose3.identity())
        self.assertEqual(result, ((1.0, 0.0, 0.0), (2.0, 0.0, 0.0)))


if __name__ == "__main__":
    unittest.main()
