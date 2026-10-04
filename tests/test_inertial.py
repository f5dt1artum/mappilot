import math
import unittest

from mappilot.geometry import Pose3
from mappilot.inertial import PreintegratedImu, preintegrate_imu


def close(actual, expected, tol=1e-9):
    return all(abs(a - e) <= tol for a, e in zip(actual, expected))


ZERO = (0.0, 0.0, 0.0)
ID_QUAT = (1.0, 0.0, 0.0, 0.0)


def rotation_z(angle):
    return (math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0))


class ZeroInputTest(unittest.TestCase):
    def test_zero_measurements_give_identity(self):
        samples = [(0.0, ZERO, ZERO), (1.0, ZERO, ZERO), (2.0, ZERO, ZERO)]
        result = preintegrate_imu(samples)
        self.assertIsInstance(result, PreintegratedImu)
        self.assertEqual(result.delta_pose, Pose3.identity())
        self.assertEqual(result.delta_velocity, ZERO)
        self.assertEqual(result.duration, 2.0)
        self.assertEqual(result.intervals, 2)

    def test_defaults_are_zero_biases(self):
        samples = [(0.0, ZERO, ZERO), (0.5, ZERO, ZERO)]
        result = preintegrate_imu(iter(samples))
        self.assertEqual(result.delta_pose, Pose3.identity())
        self.assertEqual(result.delta_velocity, ZERO)


class TranslationTest(unittest.TestCase):
    def test_constant_acceleration_without_rotation(self):
        samples = [
            (0.0, (2.0, 0.0, 0.0), ZERO),
            (1.0, (2.0, 0.0, 0.0), ZERO),
            (2.0, (2.0, 0.0, 0.0), ZERO),
        ]
        result = preintegrate_imu(samples)
        self.assertTrue(close(result.delta_pose.translation, (4.0, 0.0, 0.0)))
        self.assertTrue(close(result.delta_velocity, (4.0, 0.0, 0.0)))
        self.assertTrue(close(result.delta_pose.quaternion, ID_QUAT))
        self.assertEqual(result.duration, 2.0)
        self.assertEqual(result.intervals, 2)

    def test_uneven_sampling_uses_each_interval_length(self):
        # Same rule, irregular spacing: 0.5 s then 1.5 s, a = (0, 2, 0).
        samples = [
            (0.0, (0.0, 2.0, 0.0), ZERO),
            (0.5, (0.0, 2.0, 0.0), ZERO),
            (2.0, (0.0, 2.0, 0.0), ZERO),
        ]
        result = preintegrate_imu(samples)
        # Interval 1: dv=(0,1,0), dp=(0,0.25,0)
        # Interval 2: dv=(0,4,0), dp=(0,0.25+1.5+2.25,0)=(0,4,0)
        self.assertTrue(close(result.delta_velocity, (0.0, 4.0, 0.0)))
        self.assertTrue(close(result.delta_pose.translation, (0.0, 4.0, 0.0)))
        self.assertEqual(result.duration, 2.0)
        self.assertEqual(result.intervals, 2)


class RotationTest(unittest.TestCase):
    def test_constant_angular_rate_matches_exp(self):
        omega = (0.0, 0.0, math.pi / 2.0)
        samples = [(0.0, ZERO, omega), (1.0, ZERO, omega)]
        result = preintegrate_imu(samples)
        expected = Pose3.exp((omega[0], omega[1], omega[2], 0.0, 0.0, 0.0))
        self.assertTrue(
            close(result.delta_pose.quaternion, expected.quaternion),
            result.delta_pose.quaternion,
        )
        self.assertTrue(close(result.delta_pose.translation, ZERO))
        self.assertTrue(close(result.delta_velocity, ZERO))

    def test_specific_force_rotated_into_start_frame(self):
        samples = [
            (0.0, (0.0, 1.0, 0.0), (0.0, 0.0, math.pi / 2.0)),
            (1.0, (0.0, 1.0, 0.0), (0.0, 0.0, math.pi / 2.0)),
        ]
        result = preintegrate_imu(samples)
        # Left sample is integrated with identity rotation.
        self.assertTrue(close(result.delta_pose.translation, (0.0, 0.5, 0.0)))
        self.assertTrue(close(result.delta_velocity, (0.0, 1.0, 0.0)))
        self.assertTrue(close(result.delta_pose.quaternion, rotation_z(math.pi / 2.0)))

    def test_accumulated_rotation_steers_second_interval(self):
        samples = [
            (0.0, (0.0, 1.0, 0.0), (0.0, 0.0, math.pi)),
            (1.0, (0.0, 1.0, 0.0), ZERO),
            (2.0, (0.0, 0.0, 0.0), ZERO),
        ]
        result = preintegrate_imu(samples)
        # Interval 1: dv=(0,1,0), dp=(0,0.5,0), dR=Rz(pi).
        # Interval 2: rotated a=(0,-1,0), dv back to zero,
        # dp=(0,0.5)+(0,1)+(0,-0.5)=(0,1,0).
        self.assertTrue(close(result.delta_velocity, ZERO, tol=1e-9))
        self.assertTrue(close(result.delta_pose.translation, (0.0, 1.0, 0.0)))
        self.assertTrue(close(result.delta_pose.quaternion, rotation_z(math.pi)))


class BiasTest(unittest.TestCase):
    def test_biases_subtracted_from_left_sample_measurement(self):
        samples = [
            (0.0, (0.0, 1.0, 0.0), (0.0, 0.0, 0.1)),
            (1.0, (0.0, 1.0, 0.0), (0.0, 0.0, 0.1)),
            (2.0, (0.0, 1.0, 0.0), (0.0, 0.0, 0.1)),
        ]
        result = preintegrate_imu(
            samples,
            accelerometer_bias=(0.0, 1.0, 0.0),
            gyroscope_bias=(0.0, 0.0, 0.1),
        )
        self.assertEqual(result.delta_pose, Pose3.identity())
        self.assertEqual(result.delta_velocity, ZERO)

    def test_biases_constant_over_window(self):
        samples = [
            (0.0, (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
            (0.5, (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
            (1.0, (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        ]
        result = preintegrate_imu(
            samples, (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)
        )
        self.assertEqual(result.delta_pose, Pose3.identity())
        self.assertEqual(result.delta_velocity, ZERO)


class MaxIntervalTest(unittest.TestCase):
    SAMPLES = [(0.0, ZERO, ZERO), (1.0, ZERO, ZERO), (3.0, ZERO, ZERO)]

    def test_interval_equal_to_limit_accepted(self):
        result = preintegrate_imu(
            [(0.0, ZERO, ZERO), (2.0, ZERO, ZERO)], max_interval=2.0
        )
        self.assertEqual(result.duration, 2.0)

    def test_interval_above_limit_rejected(self):
        with self.assertRaises(ValueError):
            preintegrate_imu(self.SAMPLES, max_interval=1.5)

    def test_none_disables_limit(self):
        result = preintegrate_imu(self.SAMPLES, max_interval=None)
        self.assertEqual(result.intervals, 2)

    def test_non_positive_limit_rejected(self):
        for bad in (0.0, -0.1):
            with self.assertRaises(ValueError):
                preintegrate_imu(self.SAMPLES, max_interval=bad)

    def test_non_finite_limit_rejected(self):
        for bad in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                preintegrate_imu(self.SAMPLES, max_interval=bad)

    def test_wrong_type_limit_rejected(self):
        with self.assertRaises(TypeError):
            preintegrate_imu(self.SAMPLES, max_interval=True)
        with self.assertRaises(TypeError):
            preintegrate_imu(self.SAMPLES, max_interval="1.0")


class ValidationTest(unittest.TestCase):
    SAMPLES = [(0.0, ZERO, ZERO), (1.0, ZERO, ZERO)]

    def test_too_few_samples_raises_value_error(self):
        with self.assertRaises(ValueError):
            preintegrate_imu([(0.0, ZERO, ZERO)])
        with self.assertRaises(ValueError):
            preintegrate_imu([])

    def test_duplicate_and_reversed_timestamps_raise(self):
        with self.assertRaises(ValueError):
            preintegrate_imu([(1.0, ZERO, ZERO), (1.0, ZERO, ZERO)])
        with self.assertRaises(ValueError):
            preintegrate_imu([(2.0, ZERO, ZERO), (1.0, ZERO, ZERO)])

    def test_non_finite_values_raise_value_error(self):
        with self.assertRaises(ValueError):
            preintegrate_imu([(float("nan"), ZERO, ZERO), (1.0, ZERO, ZERO)])
        with self.assertRaises(ValueError):
            preintegrate_imu(
                [(0.0, (float("inf"), 0.0, 0.0), ZERO), (1.0, ZERO, ZERO)]
            )
        with self.assertRaises(ValueError):
            preintegrate_imu(
                [(0.0, ZERO, (0.0, float("-inf"), 0.0)), (1.0, ZERO, ZERO)]
            )
        with self.assertRaises(ValueError):
            preintegrate_imu(self.SAMPLES, accelerometer_bias=(0.0, float("nan"), 0.0))
        with self.assertRaises(ValueError):
            preintegrate_imu(self.SAMPLES, gyroscope_bias=(0.0, 0.0, float("inf")))

    def test_non_finite_accumulation_raises_value_error(self):
        huge = (1e308, 0.0, 0.0)
        with self.assertRaises(ValueError):
            preintegrate_imu([(0.0, huge, ZERO), (2.0, huge, ZERO)])

    def test_type_errors(self):
        with self.assertRaises(TypeError):
            preintegrate_imu(42)
        with self.assertRaises(TypeError):
            preintegrate_imu(None)
        with self.assertRaises(TypeError):
            preintegrate_imu([1234, 5678])
        with self.assertRaises(TypeError):
            preintegrate_imu([(0.0, ZERO), (1.0, ZERO, ZERO)])
        with self.assertRaises(TypeError):
            preintegrate_imu(
                [(0.0, ZERO, ZERO, ZERO), (1.0, ZERO, ZERO)]
            )
        with self.assertRaises(TypeError):
            preintegrate_imu(
                [(0.0, (0.0, 0.0), ZERO), (1.0, ZERO, ZERO)]
            )
        with self.assertRaises(TypeError):
            preintegrate_imu(
                [(0.0, ZERO, (0.0, 0.0, 0.0, 0.0)), (1.0, ZERO, ZERO)]
            )
        with self.assertRaises(TypeError):
            preintegrate_imu(
                [(True, ZERO, ZERO), (1.0, ZERO, ZERO)]
            )
        with self.assertRaises(TypeError):
            preintegrate_imu(
                [(0.0, (True, 0.0, 0.0), ZERO), (1.0, ZERO, ZERO)]
            )
        with self.assertRaises(TypeError):
            preintegrate_imu(
                [(0.0, ZERO, (0.0, "1", 0.0)), (1.0, ZERO, ZERO)]
            )
        with self.assertRaises(TypeError):
            preintegrate_imu(self.SAMPLES, accelerometer_bias=(0.0, 0.0))
        with self.assertRaises(TypeError):
            preintegrate_imu(self.SAMPLES, gyroscope_bias=1.5)
        with self.assertRaises(TypeError):
            preintegrate_imu(self.SAMPLES, accelerometer_bias=(True, 0.0, 0.0))
        with self.assertRaises(TypeError):
            preintegrate_imu("not-iterable")


class ContractTest(unittest.TestCase):
    def test_generator_consumed_once_and_inputs_unchanged(self):
        source = [
            (0.0, [1.0, 0.0, 0.0], [0.0, 0.0, 0.1]),
            (0.5, [1.0, 0.0, 0.0], [0.0, 0.0, 0.1]),
        ]
        consumed = []

        def generator():
            for sample in source:
                consumed.append(sample)
                yield sample

        snapshot = [
            (sample[0], list(sample[1]), list(sample[2])) for sample in source
        ]
        result = preintegrate_imu(generator())
        self.assertEqual(len(consumed), 2)
        self.assertEqual(result.intervals, 1)
        self.assertEqual(
            [(sample[0], list(sample[1]), list(sample[2])) for sample in source],
            snapshot,
        )

    def test_result_is_immutable(self):
        result = preintegrate_imu(
            [(0.0, ZERO, ZERO), (1.0, ZERO, ZERO)]
        )
        with self.assertRaises(AttributeError):
            result.duration = 5.0
        with self.assertRaises(TypeError):
            result.delta_velocity[0] = 9.0
        self.assertIsInstance(result.delta_velocity, tuple)
        self.assertIsInstance(result.delta_pose.translation, tuple)

    def test_deterministic_across_calls(self):
        samples = [
            (0.1 * i, (0.3 * i - 1.0, 0.5, -0.2 * i), (0.01, 0.02 * i, -0.01))
            for i in range(8)
        ]
        first = preintegrate_imu(samples, (0.01, 0.0, 0.02), (0.0, 0.001, 0.0))
        second = preintegrate_imu(samples, (0.01, 0.0, 0.02), (0.0, 0.001, 0.0))
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
