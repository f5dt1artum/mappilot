import math
import unittest

from mappilot.geometry import Pose3
from mappilot.inertial import ImuPreintegration, preintegrate_imu


def close(actual, expected, tol=1e-9):
    return all(abs(a - e) <= tol for a, e in zip(actual, expected))


class ZeroMeasurementTest(unittest.TestCase):
    def test_zero_measurements_yield_identity(self):
        samples = [
            (0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            (0.5, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            (1.5, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ]
        result = preintegrate_imu(samples)
        self.assertIsInstance(result, ImuPreintegration)
        self.assertEqual(result.delta_pose, Pose3.identity())
        self.assertEqual(result.delta_velocity, (0.0, 0.0, 0.0))
        self.assertEqual(result.duration, 1.5)
        self.assertEqual(result.intervals, 2)

    def test_measurement_equal_to_bias_cancels(self):
        samples = [
            (1.0, (1.0, 2.0, 3.0), (0.1, 0.2, 0.3)),
            (2.0, (1.0, 2.0, 3.0), (0.1, 0.2, 0.3)),
        ]
        result = preintegrate_imu(
            samples,
            accelerometer_bias=(1.0, 2.0, 3.0),
            gyroscope_bias=(0.1, 0.2, 0.3),
        )
        self.assertEqual(result.delta_pose, Pose3.identity())
        self.assertEqual(result.delta_velocity, (0.0, 0.0, 0.0))


class ConstantAccelerationTest(unittest.TestCase):
    def test_constant_force_no_rotation(self):
        samples = [
            (0.0, (2.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            (0.5, (2.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            (1.0, (2.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ]
        result = preintegrate_imu(samples)
        # dp = 0.5*a*T^2, dv = a*T with T = 1.
        self.assertTrue(close(result.delta_pose.translation, (1.0, 0.0, 0.0)))
        self.assertTrue(close(result.delta_velocity, (2.0, 0.0, 0.0)))
        self.assertEqual(result.delta_pose.quaternion, (1.0, 0.0, 0.0, 0.0))

    def test_bias_subtracted_before_integration(self):
        samples = [
            (0.0, (3.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            (1.0, (3.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ]
        result = preintegrate_imu(samples, accelerometer_bias=(1.0, 0.0, 0.0))
        self.assertTrue(close(result.delta_pose.translation, (1.0, 0.0, 0.0)))
        self.assertTrue(close(result.delta_velocity, (2.0, 0.0, 0.0)))

    def test_nonuniform_sampling(self):
        samples = [
            (0.0, (2.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            (0.1, (2.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            (0.7, (2.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            (1.0, (2.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ]
        result = preintegrate_imu(samples)
        # Constant force: same closed form regardless of spacing.
        self.assertTrue(close(result.delta_pose.translation, (1.0, 0.0, 0.0)))
        self.assertTrue(close(result.delta_velocity, (2.0, 0.0, 0.0)))
        self.assertEqual(result.intervals, 3)


class RotationTest(unittest.TestCase):
    def test_constant_yaw_rate(self):
        omega = 0.3
        samples = [
            (0.0, (0.0, 0.0, 0.0), (0.0, 0.0, omega)),
            (1.0, (0.0, 0.0, 0.0), (0.0, 0.0, omega)),
            (2.0, (0.0, 0.0, 0.0), (0.0, 0.0, omega)),
        ]
        result = preintegrate_imu(samples)
        angle = 2.0 * omega
        expected = (math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0))
        self.assertTrue(close(result.delta_pose.quaternion, expected))
        self.assertEqual(result.delta_pose.translation, (0.0, 0.0, 0.0))

    def test_force_rotated_into_start_frame(self):
        # Quarter turn about z during the first interval; the body-x force
        # of the second interval points along start-frame y.
        omega = math.pi / 2.0
        samples = [
            (0.0, (1.0, 0.0, 0.0), (0.0, 0.0, omega)),
            (1.0, (1.0, 0.0, 0.0), (0.0, 0.0, omega)),
            (2.0, (1.0, 0.0, 0.0), (0.0, 0.0, omega)),
        ]
        result = preintegrate_imu(samples)
        self.assertTrue(
            close(result.delta_pose.translation, (1.5, 0.5, 0.0)),
            result.delta_pose.translation,
        )
        self.assertTrue(
            close(result.delta_velocity, (1.0, 1.0, 0.0)), result.delta_velocity
        )
        # Total rotation is pi about z: canonical quaternion (0, 0, 0, 1).
        self.assertTrue(
            close(result.delta_pose.quaternion, (0.0, 0.0, 0.0, 1.0)),
            result.delta_pose.quaternion,
        )

    def test_gyroscope_bias_changes_rotation(self):
        samples = [
            (0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
            (1.0, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        ]
        result = preintegrate_imu(samples, gyroscope_bias=(0.0, 0.0, 0.25))
        angle = 0.75
        expected = (math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0))
        self.assertTrue(close(result.delta_pose.quaternion, expected))


class InterfaceContractTest(unittest.TestCase):
    def test_generator_consumed_once(self):
        def source():
            yield (0.0, (1.0, 0.0, 0.0), (0.0, 0.0, 0.0))
            yield (1.0, (1.0, 0.0, 0.0), (0.0, 0.0, 0.0))

        result = preintegrate_imu(source())
        self.assertTrue(close(result.delta_pose.translation, (0.5, 0.0, 0.0)))

    def test_input_not_modified(self):
        accel = [1.0, 0.0, 0.0]
        gyro = [0.0, 0.0, 0.1]
        samples = [[0.0, accel, gyro], [1.0, list(accel), list(gyro)]]
        accel_bias = [0.1, 0.0, 0.0]
        gyro_bias = [0.0, 0.0, 0.1]
        preintegrate_imu(
            samples, accelerometer_bias=accel_bias, gyroscope_bias=gyro_bias
        )
        self.assertEqual(accel, [1.0, 0.0, 0.0])
        self.assertEqual(gyro, [0.0, 0.0, 0.1])
        self.assertEqual(accel_bias, [0.1, 0.0, 0.0])
        self.assertEqual(gyro_bias, [0.0, 0.0, 0.1])
        self.assertEqual(samples[0][0], 0.0)

    def test_same_inputs_give_equal_results(self):
        samples = [
            (0.0, (1.0, 2.0, 3.0), (0.1, 0.2, 0.3)),
            (0.3, (1.1, 1.9, 3.1), (0.1, 0.2, 0.4)),
            (1.0, (0.9, 2.1, 2.9), (0.2, 0.2, 0.3)),
        ]
        first = preintegrate_imu(iter(samples))
        second = preintegrate_imu(iter(samples))
        self.assertEqual(first, second)

    def test_result_is_immutable(self):
        result = preintegrate_imu(
            [(0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
             (1.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))]
        )
        with self.assertRaises(AttributeError):
            result.duration = 5.0
        self.assertIsInstance(result.delta_pose, Pose3)
        self.assertIsInstance(result.delta_velocity, tuple)


class TypeErrorTest(unittest.TestCase):
    def test_non_iterable_samples(self):
        with self.assertRaises(TypeError):
            preintegrate_imu(42)
        with self.assertRaises(TypeError):
            preintegrate_imu("samples")

    def test_malformed_record(self):
        good = (0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        with self.assertRaises(TypeError):
            preintegrate_imu([good, 7])
        with self.assertRaises(TypeError):
            preintegrate_imu([good, (1.0, (0.0, 0.0, 0.0))])
        with self.assertRaises(TypeError):
            preintegrate_imu([good, (1.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1)])

    def test_wrong_vector_dimension(self):
        good = (0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        with self.assertRaises(TypeError):
            preintegrate_imu([good, (1.0, (0.0, 0.0), (0.0, 0.0, 0.0))])
        with self.assertRaises(TypeError):
            preintegrate_imu([good, (1.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0))])
        with self.assertRaises(TypeError):
            preintegrate_imu([good, (1.0, "abc", (0.0, 0.0, 0.0))])

    def test_non_real_scalars(self):
        good = (0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        with self.assertRaises(TypeError):
            preintegrate_imu([good, (True, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))])
        with self.assertRaises(TypeError):
            preintegrate_imu([good, (1.0, (0.0, None, 0.0), (0.0, 0.0, 0.0))])
        with self.assertRaises(TypeError):
            preintegrate_imu([good, (1.0, (0.0, 0.0, 0.0), (0.0, 0.0, 1 + 2j))])

    def test_non_real_bias(self):
        samples = [
            (0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            (1.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ]
        with self.assertRaises(TypeError):
            preintegrate_imu(samples, accelerometer_bias=(0.0, True, 0.0))
        with self.assertRaises(TypeError):
            preintegrate_imu(samples, gyroscope_bias=(0.0, 0.0))
        with self.assertRaises(TypeError):
            preintegrate_imu(samples, gyroscope_bias=1.5)

    def test_bad_max_interval_type(self):
        samples = [
            (0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            (1.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ]
        with self.assertRaises(TypeError):
            preintegrate_imu(samples, max_interval="1.0")
        with self.assertRaises(TypeError):
            preintegrate_imu(samples, max_interval=True)


class ValueErrorTest(unittest.TestCase):
    def test_too_few_samples(self):
        with self.assertRaises(ValueError):
            preintegrate_imu([])
        with self.assertRaises(ValueError):
            preintegrate_imu([(0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))])

    def test_non_finite_values(self):
        good = (0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        with self.assertRaises(ValueError):
            preintegrate_imu([good, (math.nan, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))])
        with self.assertRaises(ValueError):
            preintegrate_imu([good, (1.0, (math.inf, 0.0, 0.0), (0.0, 0.0, 0.0))])
        with self.assertRaises(ValueError):
            preintegrate_imu([good, (1.0, (0.0, 0.0, 0.0), (0.0, -math.inf, 0.0))])
        with self.assertRaises(ValueError):
            preintegrate_imu([good, good], accelerometer_bias=(math.nan, 0.0, 0.0))

    def test_timestamps_not_increasing(self):
        record = (0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        with self.assertRaises(ValueError):
            preintegrate_imu([(1.0,) + record[1:], (1.0,) + record[1:]])
        with self.assertRaises(ValueError):
            preintegrate_imu([(2.0,) + record[1:], (1.0,) + record[1:]])

    def test_bad_max_interval_value(self):
        samples = [
            (0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            (1.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ]
        for bad in (0.0, -1.0, math.nan, math.inf):
            with self.assertRaises(ValueError):
                preintegrate_imu(samples, max_interval=bad)

    def test_interval_limit_enforced_strictly(self):
        samples = [
            (0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            (0.5, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            (1.5, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ]
        with self.assertRaises(ValueError):
            preintegrate_imu(samples, max_interval=0.5)
        # An interval exactly equal to the limit is accepted.
        result = preintegrate_imu(samples[:2], max_interval=0.5)
        self.assertEqual(result.duration, 0.5)

    def test_non_finite_accumulation_rejected(self):
        samples = [
            (0.0, (1e308, 0.0, 0.0), (0.0, 0.0, 0.0)),
            (2.0, (1e308, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ]
        with self.assertRaises(ValueError):
            preintegrate_imu(samples)


if __name__ == "__main__":
    unittest.main()
