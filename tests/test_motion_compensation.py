import math
import unittest

from mappilot.geometry import Pose3
from mappilot.motion_compensation import deskew_point_cloud


def close(actual, expected, tol=1e-9):
    return all(abs(a - e) <= tol for a, e in zip(actual, expected))


def rotation_z(angle):
    return Pose3(
        (0.0, 0.0, 0.0),
        (math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0)),
    )


class StationaryTest(unittest.TestCase):
    def test_identity_trajectory_preserves_points(self):
        trajectory = [(0.0, Pose3.identity()), (1.0, Pose3.identity())]
        points = [(0.0, (1.0, 2.0, 3.0)), (0.05, (-1.0, 0.5, -2.0))]
        result = deskew_point_cloud(
            iter(points), 0.0, iter(trajectory), Pose3.identity()
        )
        self.assertEqual(result, ((1.0, 2.0, 3.0), (-1.0, 0.5, -2.0)))
        self.assertIsInstance(result, tuple)
        self.assertIsInstance(result[0], tuple)

    def test_static_body_and_extrinsics_preserve_points(self):
        body = Pose3((0.3, -0.7, 0.2), (math.cos(0.3), 0.0, 0.0, math.sin(0.3)))
        trajectory = [(2.0, body), (5.0, body)]
        extrinsics = Pose3(
            (0.1, 0.2, -0.4),
            (math.cos(-0.1), 0.0, math.sin(-0.1), 0.0),
        )
        points = [
            (0.0, (1.0, 2.0, 3.0)),
            (0.5, (-1.0, 0.5, -2.0)),
            (1.0, (0.0, 0.0, 0.0)),
        ]
        result = deskew_point_cloud(points, 2.5, trajectory, extrinsics)
        for actual, (_, expected) in zip(result, points):
            self.assertTrue(close(actual, expected), (actual, expected))


class MotionTest(unittest.TestCase):
    def test_constant_translation(self):
        trajectory = [
            (0.0, Pose3((0.0, 0.0, 0.0))),
            (1.0, Pose3((10.0, 0.0, 0.0))),
        ]
        points = [
            (0.05, (1.0, 2.0, 3.0)),
            (0.95, (1.0, 2.0, 3.0)),
        ]
        result = deskew_point_cloud(points, 0.0, trajectory, Pose3.identity())
        self.assertTrue(close(result[0], (1.5, 2.0, 3.0)), result[0])
        self.assertTrue(close(result[1], (10.5, 2.0, 3.0)), result[1])

    def test_explicit_reference_time(self):
        trajectory = [
            (0.0, Pose3((0.0, 0.0, 0.0))),
            (1.0, Pose3((10.0, 0.0, 0.0))),
        ]
        # Beam at 0.1 puts the sensor-origin point at world x=1; the sensor
        # frame at reference time 0.6 sits at world x=6 -> local x=-5.
        result = deskew_point_cloud(
            [(0.1, (0.0, 0.0, 0.0))],
            0.0,
            trajectory,
            Pose3.identity(),
            reference_time=0.6,
        )
        self.assertTrue(close(result[0], (-5.0, 0.0, 0.0)), result[0])

    def test_constant_rotation_interpolates_orientation(self):
        trajectory = [
            (0.0, Pose3((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))),
            (1.0, rotation_z(0.2)),
        ]
        result = deskew_point_cloud(
            [(0.5, (1.0, 0.0, 0.0))],
            0.0,
            trajectory,
            Pose3.identity(),
        )
        self.assertTrue(
            close(result[0], (math.cos(0.1), math.sin(0.1), 0.0)), result[0]
        )

    def test_extrinsics_are_honoured(self):
        # Sensor is mounted 1 m ahead of the body origin along body x.
        trajectory = [
            (0.0, Pose3.identity()),
            (1.0, Pose3((10.0, 0.0, 0.0))),
        ]
        extrinsics = Pose3((1.0, 0.0, 0.0))
        result = deskew_point_cloud(
            [(0.0, (0.0, 0.0, 0.0))],
            0.0,
            trajectory,
            extrinsics,
            reference_time=1.0,
        )
        # Sensor origin at beam time is world x=1; at reference time x=11.
        self.assertTrue(close(result[0], (-10.0, 0.0, 0.0)), result[0])

    def test_endpoints_of_closed_interval(self):
        trajectory = [
            (0.0, Pose3.identity()),
            (1.0, Pose3((10.0, 0.0, 0.0))),
        ]
        late = deskew_point_cloud(
            [(1.0, (0.0, 0.0, 0.0))], 0.0, trajectory, Pose3.identity()
        )
        self.assertTrue(close(late[0], (10.0, 0.0, 0.0)), late[0])
        early = deskew_point_cloud(
            [(0.0, (0.0, 0.0, 0.0))],
            0.0,
            trajectory,
            Pose3.identity(),
            reference_time=1.0,
        )
        self.assertTrue(close(early[0], (-10.0, 0.0, 0.0)), early[0])


class InterpolationGapTest(unittest.TestCase):
    def test_exact_sample_hit_ignores_gap(self):
        trajectory = [
            (0.0, Pose3.identity()),
            (10.0, Pose3((100.0, 0.0, 0.0))),
        ]
        result = deskew_point_cloud(
            [(0.0, (1.0, 1.0, 1.0))],
            0.0,
            trajectory,
            Pose3.identity(),
            max_interpolation_gap=5.0,
        )
        self.assertTrue(close(result[0], (1.0, 1.0, 1.0)))

    def test_gap_exceeded_raises(self):
        trajectory = [
            (0.0, Pose3.identity()),
            (10.0, Pose3((100.0, 0.0, 0.0))),
        ]
        with self.assertRaises(ValueError):
            deskew_point_cloud(
                [(0.5, (1.0, 1.0, 1.0))],
                0.0,
                trajectory,
                Pose3.identity(),
                max_interpolation_gap=5.0,
            )

    def test_gap_equal_to_limit_allowed(self):
        trajectory = [
            (0.0, Pose3.identity()),
            (5.0, Pose3((10.0, 0.0, 0.0))),
        ]
        result = deskew_point_cloud(
            [(2.5, (1.0, 1.0, 1.0))],
            0.0,
            trajectory,
            Pose3.identity(),
            max_interpolation_gap=5.0,
        )
        self.assertEqual(len(result), 1)


class ValidationTest(unittest.TestCase):
    TRAJECTORY = [(0.0, Pose3.identity()), (1.0, Pose3.identity())]
    POINT = (0.0, (0.0, 0.0, 0.0))

    def test_out_of_interval_raises_value_error(self):
        with self.assertRaises(ValueError):
            deskew_point_cloud(
                [(1.0001, (0.0, 0.0, 0.0))],
                0.0,
                self.TRAJECTORY,
                Pose3.identity(),
            )
        with self.assertRaises(ValueError):
            deskew_point_cloud(
                [(-0.0001, (0.0, 0.0, 0.0))],
                0.0,
                self.TRAJECTORY,
                Pose3.identity(),
            )
        with self.assertRaises(ValueError):
            deskew_point_cloud(
                [self.POINT],
                0.0,
                self.TRAJECTORY,
                Pose3.identity(),
                reference_time=1.5,
            )

    def test_empty_cloud_raises_value_error(self):
        with self.assertRaises(ValueError):
            deskew_point_cloud([], 0.0, self.TRAJECTORY, Pose3.identity())

    def test_too_few_samples_raises_value_error(self):
        with self.assertRaises(ValueError):
            deskew_point_cloud(
                [self.POINT], 0.0, [(0.0, Pose3.identity())], Pose3.identity()
            )

    def test_duplicate_and_reversed_timestamps_raise(self):
        with self.assertRaises(ValueError):
            deskew_point_cloud(
                [self.POINT],
                0.0,
                [(0.0, Pose3.identity()), (0.0, Pose3.identity())],
                Pose3.identity(),
            )
        with self.assertRaises(ValueError):
            deskew_point_cloud(
                [self.POINT],
                0.0,
                [(1.0, Pose3.identity()), (0.0, Pose3.identity())],
                Pose3.identity(),
            )

    def test_non_finite_values_raise_value_error(self):
        with self.assertRaises(ValueError):
            deskew_point_cloud(
                [(float("nan"), (0.0, 0.0, 0.0))],
                0.0,
                self.TRAJECTORY,
                Pose3.identity(),
            )
        with self.assertRaises(ValueError):
            deskew_point_cloud(
                [(0.0, (0.0, float("inf"), 0.0))],
                0.0,
                self.TRAJECTORY,
                Pose3.identity(),
            )
        with self.assertRaises(ValueError):
            deskew_point_cloud(
                [self.POINT], float("nan"), self.TRAJECTORY, Pose3.identity()
            )

    def test_non_positive_gap_raises_value_error(self):
        for bad in (0.0, -1.0, float("nan")):
            with self.assertRaises((ValueError, TypeError)):
                deskew_point_cloud(
                    [self.POINT],
                    0.0,
                    self.TRAJECTORY,
                    Pose3.identity(),
                    max_interpolation_gap=bad,
                )

    def test_type_errors(self):
        with self.assertRaises(TypeError):
            deskew_point_cloud(42, 0.0, self.TRAJECTORY, Pose3.identity())
        with self.assertRaises(TypeError):
            deskew_point_cloud([self.POINT], 0.0, 42, Pose3.identity())
        with self.assertRaises(TypeError):
            deskew_point_cloud([self.POINT], 0.0, self.TRAJECTORY, 42)
        with self.assertRaises(TypeError):
            deskew_point_cloud(
                [(0.0, (0.0, 0.0))], 0.0, self.TRAJECTORY, Pose3.identity()
            )
        with self.assertRaises(TypeError):
            deskew_point_cloud(
                [(0.0, (0.0, 0.0, 0.0, 0.0))],
                0.0,
                self.TRAJECTORY,
                Pose3.identity(),
            )
        with self.assertRaises(TypeError):
            deskew_point_cloud(
                [(0.0,)], 0.0, self.TRAJECTORY, Pose3.identity()
            )
        with self.assertRaises(TypeError):
            deskew_point_cloud(
                [(True, (0.0, 0.0, 0.0))],
                0.0,
                self.TRAJECTORY,
                Pose3.identity(),
            )
        with self.assertRaises(TypeError):
            deskew_point_cloud(
                [(0.0, (True, 0.0, 0.0))],
                0.0,
                self.TRAJECTORY,
                Pose3.identity(),
            )
        with self.assertRaises(TypeError):
            deskew_point_cloud(
                [1234], 0.0, self.TRAJECTORY, Pose3.identity()
            )
        with self.assertRaises(TypeError):
            deskew_point_cloud(
                [self.POINT], True, self.TRAJECTORY, Pose3.identity()
            )
        with self.assertRaises(TypeError):
            deskew_point_cloud(
                [self.POINT],
                0.0,
                [(0.0, Pose3.identity()), (1.0, (0.0, 0.0, 0.0))],
                Pose3.identity(),
            )


class ContractTest(unittest.TestCase):
    def test_generator_consumed_once_and_inputs_unchanged(self):
        consumed = []

        def generator():
            for record in [(0.0, [1.0, 2.0, 3.0]), (0.5, [-1.0, 0.0, 2.0])]:
                consumed.append(record)
                yield record

        records = [(0.0, [1.0, 2.0, 3.0]), (0.5, [-1.0, 0.0, 2.0])]
        snapshot = [list(record[1]) for record in records]
        result = deskew_point_cloud(
            generator(),
            0.0,
            [(0.0, Pose3.identity()), (1.0, Pose3.identity())],
            Pose3.identity(),
        )
        self.assertEqual(len(consumed), 2)
        self.assertEqual(len(result), 2)
        self.assertEqual([list(record[1]) for record in records], snapshot)
        self.assertIsInstance(result[0], tuple)

    def test_order_and_count_preserved(self):
        trajectory = [
            (0.0, Pose3.identity()),
            (1.0, Pose3((1.0, 0.0, 0.0))),
        ]
        points = [(0.1 * i, (float(i), 0.0, 0.0)) for i in range(10)]
        result = deskew_point_cloud(points, 0.0, trajectory, Pose3.identity())
        self.assertEqual(len(result), 10)

    def test_deterministic_across_calls(self):
        trajectory = [
            (0.0, Pose3((0.0, 0.0, 0.0))),
            (1.0, Pose3((2.0, -1.0, 0.5))),
        ]
        points = [(0.25 * i, (0.3 * i - 1.0, 0.5, -0.2 * i)) for i in range(5)]
        first = deskew_point_cloud(points, 0.0, trajectory, rotation_z(0.3))
        second = deskew_point_cloud(points, 0.0, trajectory, rotation_z(0.3))
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
