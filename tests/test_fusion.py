import math
import unittest

from mappilot.fusion import FusionState, fuse_imu_pose
from mappilot.geometry import Pose3

ZERO = (0.0, 0.0, 0.0)
ID_QUAT = (1.0, 0.0, 0.0, 0.0)
GRAVITY = (0.0, 0.0, -9.81)


def close(actual, expected, tol=1e-9):
    return all(abs(a - e) <= tol for a, e in zip(actual, expected))


def rotation_z(angle):
    return (math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0))


def stationary_samples(n=3, dt=1.0, start=0.0, accel=ZERO, omega=ZERO):
    return [
        (start + i * dt, accel, omega)
        for i in range(n)
    ]


class DeadReckoningTest(unittest.TestCase):
    def test_no_observation_zero_input_keeps_initial_state(self):
        result = fuse_imu_pose(
            stationary_samples(), [], Pose3.identity(), ZERO
        )
        self.assertEqual(len(result), 3)
        for state in result:
            self.assertIsInstance(state, FusionState)
            self.assertEqual(state.pose, Pose3.identity())
            self.assertEqual(state.velocity, ZERO)
            self.assertIsNone(state.observation_used)
        self.assertEqual([s.timestamp for s in result], [0.0, 1.0, 2.0])

    def test_constant_acceleration_translation(self):
        samples = [
            (0.0, (2.0, 0.0, 0.0), ZERO),
            (1.0, (2.0, 0.0, 0.0), ZERO),
            (2.0, (2.0, 0.0, 0.0), ZERO),
        ]
        result = fuse_imu_pose(samples, [], Pose3.identity(), ZERO)
        self.assertTrue(close(result[1].pose.translation, (1.0, 0.0, 0.0)))
        self.assertTrue(close(result[1].velocity, (2.0, 0.0, 0.0)))
        self.assertTrue(close(result[2].pose.translation, (4.0, 0.0, 0.0)))
        self.assertTrue(close(result[2].velocity, (4.0, 0.0, 0.0)))

    def test_initial_velocity_propagates(self):
        samples = stationary_samples(2, dt=2.0)
        result = fuse_imu_pose(
            samples, [], Pose3((1.0, 2.0, 3.0), ID_QUAT), (1.0, 0.0, -1.0)
        )
        self.assertTrue(close(result[0].pose.translation, (1.0, 2.0, 3.0)))
        self.assertTrue(close(result[1].pose.translation, (3.0, 2.0, 1.0)))
        self.assertEqual(result[1].velocity, (1.0, 0.0, -1.0))

    def test_gravity_added_in_world_frame(self):
        samples = stationary_samples(2, dt=1.0)
        result = fuse_imu_pose(
            samples, [], Pose3.identity(), ZERO, gravity=GRAVITY
        )
        self.assertTrue(close(result[1].velocity, (0.0, 0.0, -9.81)))
        self.assertTrue(
            close(result[1].pose.translation, (0.0, 0.0, -0.5 * 9.81))
        )

    def test_rotation_right_composed(self):
        omega = (0.0, 0.0, math.pi / 2.0)
        samples = stationary_samples(2, dt=1.0, omega=omega)
        result = fuse_imu_pose(samples, [], Pose3.identity(), ZERO)
        self.assertTrue(close(result[1].pose.quaternion, rotation_z(math.pi / 2.0)))

    def test_specific_force_rotated_with_left_end_rotation(self):
        # Interval uses R at t[k] = identity, so accel integrates along +y.
        samples = [
            (0.0, (0.0, 1.0, 0.0), (0.0, 0.0, math.pi / 2.0)),
            (1.0, (0.0, 1.0, 0.0), ZERO),
        ]
        result = fuse_imu_pose(samples, [], Pose3.identity(), ZERO)
        self.assertTrue(close(result[1].pose.translation, (0.0, 0.5, 0.0)))
        self.assertTrue(close(result[1].velocity, (0.0, 1.0, 0.0)))
        self.assertTrue(close(result[1].pose.quaternion, rotation_z(math.pi / 2.0)))

    def test_biases_subtracted_from_left_measurement(self):
        samples = [
            (0.0, (0.0, 1.0, 0.0), (0.0, 0.0, 0.1)),
            (1.0, (0.0, 1.0, 0.0), (0.0, 0.0, 0.1)),
        ]
        result = fuse_imu_pose(
            samples,
            [],
            Pose3.identity(),
            ZERO,
            accelerometer_bias=(0.0, 1.0, 0.0),
            gyroscope_bias=(0.0, 0.0, 0.1),
        )
        self.assertEqual(result[1].pose, Pose3.identity())
        self.assertEqual(result[1].velocity, ZERO)


class ObservationTest(unittest.TestCase):
    def test_full_gain_correction_snaps_to_observation(self):
        samples = stationary_samples(3, dt=1.0)
        observed = Pose3((0.0, 5.0, 0.0), ID_QUAT)
        result = fuse_imu_pose(
            samples, [(1.0, observed)], Pose3.identity(), ZERO
        )
        self.assertIsNone(result[0].observation_used)
        self.assertTrue(result[1].observation_used)
        self.assertEqual(result[1].pose, observed)
        # Velocity untouched, later propagation continues from corrected pose.
        self.assertEqual(result[1].velocity, ZERO)
        self.assertEqual(result[2].pose, observed)
        self.assertIsNone(result[2].observation_used)

    def test_partial_translation_gain(self):
        samples = stationary_samples(2)
        observed = Pose3((10.0, 0.0, 0.0), ID_QUAT)
        result = fuse_imu_pose(
            samples, [(1.0, observed)], Pose3.identity(), ZERO,
            translation_gain=0.25, rotation_gain=1.0,
        )
        self.assertTrue(close(result[1].pose.translation, (2.5, 0.0, 0.0)))

    def test_zero_gain_keeps_prediction(self):
        samples = stationary_samples(2)
        observed = Pose3((10.0, 0.0, 0.0), rotation_z(math.pi / 2.0))
        result = fuse_imu_pose(
            samples, [(1.0, observed)], Pose3.identity(), ZERO,
            translation_gain=0.0, rotation_gain=0.0,
        )
        self.assertTrue(result[1].observation_used)
        self.assertEqual(result[1].pose, Pose3.identity())

    def test_rotation_gain_scales_angle(self):
        samples = stationary_samples(2)
        observed = Pose3(ZERO, rotation_z(math.pi / 2.0))
        result = fuse_imu_pose(
            samples, [(1.0, observed)], Pose3.identity(), ZERO,
            translation_gain=1.0, rotation_gain=0.5,
        )
        self.assertTrue(close(result[1].pose.quaternion, rotation_z(math.pi / 4.0)))

    def test_observation_at_first_timestamp_corrects_initial_state(self):
        samples = stationary_samples(2)
        observed = Pose3((3.0, 0.0, 0.0), ID_QUAT)
        result = fuse_imu_pose(
            samples, [(0.0, observed)], Pose3.identity(), ZERO
        )
        self.assertTrue(result[0].observation_used)
        self.assertEqual(result[0].pose, observed)
        self.assertTrue(close(result[0].velocity, ZERO))
        # Propagation after the corrected initial pose.
        self.assertTrue(close(result[1].pose.translation, (3.0, 0.0, 0.0)))

    def test_residual_in_body_frame(self):
        # The residual T_pred^-1 * T_obs is expressed in the predicted body
        # frame: a world-frame yaw of the body rotates the world offset.
        yaw = math.pi / 2.0
        predicted_initial = Pose3(ZERO, rotation_z(yaw))
        samples = stationary_samples(2)
        # Observed world translation (5, 0, 0); in predicted body frame that
        # is R^T * (5,0,0) = (0, -5, 0). Threshold on body-frame norm = 5.
        observed = Pose3((5.0, 0.0, 0.0), rotation_z(yaw))
        accepted = fuse_imu_pose(
            samples, [(1.0, observed)], predicted_initial, ZERO,
            translation_gain=1.0, residual_threshold=5.0,
        )
        self.assertTrue(accepted[1].observation_used)
        rejected = fuse_imu_pose(
            samples, [(1.0, observed)], predicted_initial, ZERO,
            translation_gain=1.0, residual_threshold=4.999,
        )
        self.assertFalse(rejected[1].observation_used)
        # Rejection keeps the predicted state: no motion, initial pose stands.
        self.assertEqual(rejected[1].pose, predicted_initial)

    def test_threshold_equality_accepted(self):
        samples = stationary_samples(2)
        observed = Pose3((3.0, 4.0, 0.0), ID_QUAT)  # norm exactly 5
        result = fuse_imu_pose(
            samples, [(1.0, observed)], Pose3.identity(), ZERO,
            residual_threshold=5.0,
        )
        self.assertTrue(result[1].observation_used)
        self.assertEqual(result[1].pose, observed)

    def test_translation_threshold_rejection_keeps_prediction(self):
        samples = stationary_samples(3)
        observed = Pose3((0.0, 0.0, 5.0001), ID_QUAT)
        result = fuse_imu_pose(
            samples, [(1.0, observed)], Pose3.identity(), ZERO,
            residual_threshold=5.0,
        )
        self.assertFalse(result[1].observation_used)
        self.assertEqual(result[1].pose, Pose3.identity())
        self.assertIsNone(result[2].observation_used)

    def test_rotation_threshold_rejection(self):
        samples = stationary_samples(2)
        observed = Pose3(ZERO, rotation_z(0.2))
        result = fuse_imu_pose(
            samples, [(1.0, observed)], Pose3.identity(), ZERO,
            residual_threshold=0.1,
        )
        self.assertFalse(result[1].observation_used)
        self.assertEqual(result[1].pose, Pose3.identity())

    def test_accepted_then_rejected_mix(self):
        samples = stationary_samples(4)
        observations = [
            (1.0, Pose3((1.0, 0.0, 0.0), ID_QUAT)),
            (2.0, Pose3((100.0, 0.0, 0.0), ID_QUAT)),
        ]
        result = fuse_imu_pose(
            samples, observations, Pose3.identity(), ZERO,
            residual_threshold=5.0,
        )
        flags = [s.observation_used for s in result]
        self.assertEqual(flags, [None, True, False, None])
        # Rejection keeps the prediction: still at x=1 from t=1 onward.
        self.assertTrue(close(result[2].pose.translation, (1.0, 0.0, 0.0)))
        self.assertTrue(close(result[3].pose.translation, (1.0, 0.0, 0.0)))

    def test_empty_observation_sequence_allowed(self):
        result = fuse_imu_pose(stationary_samples(2), iter([]),
                               Pose3.identity(), ZERO)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(s.observation_used is None for s in result))

    def test_first_timestamp_observation_can_be_rejected(self):
        samples = stationary_samples(2)
        observed = Pose3((10.0, 0.0, 0.0), ID_QUAT)
        result = fuse_imu_pose(
            samples, [(0.0, observed)], Pose3.identity(), ZERO,
            residual_threshold=1.0,
        )
        self.assertFalse(result[0].observation_used)
        self.assertEqual(result[0].pose, Pose3.identity())


class ValidationTest(unittest.TestCase):
    SAMPLES = [(0.0, ZERO, ZERO), (1.0, ZERO, ZERO)]

    def test_too_few_samples_raises_value_error(self):
        with self.assertRaises(ValueError):
            fuse_imu_pose([(0.0, ZERO, ZERO)], [], Pose3.identity(), ZERO)
        with self.assertRaises(ValueError):
            fuse_imu_pose([], [], Pose3.identity(), ZERO)

    def test_duplicate_and_reversed_imu_timestamps_raise(self):
        with self.assertRaises(ValueError):
            fuse_imu_pose(
                [(1.0, ZERO, ZERO), (1.0, ZERO, ZERO)], [],
                Pose3.identity(), ZERO,
            )
        with self.assertRaises(ValueError):
            fuse_imu_pose(
                [(2.0, ZERO, ZERO), (1.0, ZERO, ZERO)], [],
                Pose3.identity(), ZERO,
            )

    def test_non_finite_values_raise_value_error(self):
        with self.assertRaises(ValueError):
            fuse_imu_pose(
                [(float("nan"), ZERO, ZERO), (1.0, ZERO, ZERO)], [],
                Pose3.identity(), ZERO,
            )
        with self.assertRaises(ValueError):
            fuse_imu_pose(
                [(0.0, (float("inf"), 0.0, 0.0), ZERO), (1.0, ZERO, ZERO)],
                [], Pose3.identity(), ZERO,
            )
        with self.assertRaises(ValueError):
            fuse_imu_pose(self.SAMPLES, [], Pose3.identity(),
                          (0.0, float("nan"), 0.0))
        with self.assertRaises(ValueError):
            fuse_imu_pose(self.SAMPLES, [], Pose3.identity(), ZERO,
                          gravity=(0.0, 0.0, float("inf")))

    def test_non_finite_propagation_raises_value_error(self):
        huge = (1e308, 0.0, 0.0)
        with self.assertRaises(ValueError):
            fuse_imu_pose(
                [(0.0, huge, ZERO), (2.0, huge, ZERO)], [],
                Pose3.identity(), ZERO,
            )

    def test_observation_timestamp_must_hit_imu_sample(self):
        with self.assertRaises(ValueError):
            fuse_imu_pose(
                self.SAMPLES, [(0.5, Pose3.identity())],
                Pose3.identity(), ZERO,
            )

    def test_observation_timestamps_strictly_increasing(self):
        with self.assertRaises(ValueError):
            fuse_imu_pose(
                stationary_samples(3),
                [(1.0, Pose3.identity()), (1.0, Pose3.identity())],
                Pose3.identity(), ZERO,
            )
        with self.assertRaises(ValueError):
            fuse_imu_pose(
                stationary_samples(3),
                [(2.0, Pose3.identity()), (1.0, Pose3.identity())],
                Pose3.identity(), ZERO,
            )

    def test_gain_out_of_range_raises_value_error(self):
        for kwargs in (
            {"translation_gain": -0.01},
            {"translation_gain": 1.01},
            {"rotation_gain": -1.0},
            {"rotation_gain": 2.0},
        ):
            with self.assertRaises(ValueError):
                fuse_imu_pose(self.SAMPLES, [], Pose3.identity(), ZERO, **kwargs)

    def test_non_finite_gain_raises_value_error(self):
        with self.assertRaises(ValueError):
            fuse_imu_pose(self.SAMPLES, [], Pose3.identity(), ZERO,
                          translation_gain=float("nan"))
        with self.assertRaises(ValueError):
            fuse_imu_pose(self.SAMPLES, [], Pose3.identity(), ZERO,
                          rotation_gain=float("inf"))

    def test_boolean_gain_raises_type_error(self):
        with self.assertRaises(TypeError):
            fuse_imu_pose(self.SAMPLES, [], Pose3.identity(), ZERO,
                          translation_gain=True)

    def test_threshold_negative_or_non_finite_raises_value_error(self):
        for bad in (-0.001, float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                fuse_imu_pose(self.SAMPLES, [], Pose3.identity(), ZERO,
                              residual_threshold=bad)

    def test_threshold_wrong_type_raises_type_error(self):
        with self.assertRaises(TypeError):
            fuse_imu_pose(self.SAMPLES, [], Pose3.identity(), ZERO,
                          residual_threshold="1.0")
        with self.assertRaises(TypeError):
            fuse_imu_pose(self.SAMPLES, [], Pose3.identity(), ZERO,
                          residual_threshold=True)

    def test_zero_threshold_is_valid(self):
        # Zero threshold accepts only an exactly matching observation.
        result = fuse_imu_pose(
            self.SAMPLES, [(1.0, Pose3.identity())],
            Pose3.identity(), ZERO, residual_threshold=0.0,
        )
        self.assertTrue(result[1].observation_used)

    def test_initial_pose_wrong_type_raises_type_error(self):
        with self.assertRaises(TypeError):
            fuse_imu_pose(self.SAMPLES, [], ((0.0, 0.0, 0.0), ID_QUAT), ZERO)
        with self.assertRaises(TypeError):
            fuse_imu_pose(self.SAMPLES, [], "pose", ZERO)

    def test_observation_pose_wrong_type_raises_type_error(self):
        with self.assertRaises(TypeError):
            fuse_imu_pose(
                self.SAMPLES, [(0.0, (ZERO, ID_QUAT))],
                Pose3.identity(), ZERO,
            )
        with self.assertRaises(TypeError):
            fuse_imu_pose(
                self.SAMPLES, [(0.0, 42)], Pose3.identity(), ZERO
            )

    def test_sample_shape_errors_raise_type_error(self):
        with self.assertRaises(TypeError):
            fuse_imu_pose(42, [], Pose3.identity(), ZERO)
        with self.assertRaises(TypeError):
            fuse_imu_pose(None, [], Pose3.identity(), ZERO)
        with self.assertRaises(TypeError):
            fuse_imu_pose([1234, 5678], [], Pose3.identity(), ZERO)
        with self.assertRaises(TypeError):
            fuse_imu_pose(
                [(0.0, ZERO), (1.0, ZERO, ZERO)], [],
                Pose3.identity(), ZERO,
            )
        with self.assertRaises(TypeError):
            fuse_imu_pose(
                [(0.0, (0.0, 0.0), ZERO), (1.0, ZERO, ZERO)], [],
                Pose3.identity(), ZERO,
            )
        with self.assertRaises(TypeError):
            fuse_imu_pose(
                [(True, ZERO, ZERO), (1.0, ZERO, ZERO)], [],
                Pose3.identity(), ZERO,
            )
        with self.assertRaises(TypeError):
            fuse_imu_pose(
                [(0.0, (True, 0.0, 0.0), ZERO), (1.0, ZERO, ZERO)],
                [], Pose3.identity(), ZERO,
            )
        with self.assertRaises(TypeError):
            fuse_imu_pose("not-iterable", [], Pose3.identity(), ZERO)

    def test_observation_shape_errors_raise_type_error(self):
        with self.assertRaises(TypeError):
            fuse_imu_pose(self.SAMPLES, 42, Pose3.identity(), ZERO)
        with self.assertRaises(TypeError):
            fuse_imu_pose(
                self.SAMPLES, [(1.0,)], Pose3.identity(), ZERO
            )
        with self.assertRaises(TypeError):
            fuse_imu_pose(
                self.SAMPLES,
                [(1.0, Pose3.identity(), Pose3.identity())],
                Pose3.identity(), ZERO,
            )
        with self.assertRaises(TypeError):
            fuse_imu_pose(
                self.SAMPLES, [("1.0", Pose3.identity())],
                Pose3.identity(), ZERO,
            )

    def test_bias_and_velocity_dimension_errors_raise_type_error(self):
        with self.assertRaises(TypeError):
            fuse_imu_pose(self.SAMPLES, [], Pose3.identity(), (0.0, 0.0))
        with self.assertRaises(TypeError):
            fuse_imu_pose(self.SAMPLES, [], Pose3.identity(), ZERO,
                          accelerometer_bias=1.5)
        with self.assertRaises(TypeError):
            fuse_imu_pose(self.SAMPLES, [], Pose3.identity(), ZERO,
                          gyroscope_bias=(0.0, 0.0, 0.0, 0.0))


class ContractTest(unittest.TestCase):
    def test_generators_consumed_once_and_inputs_unchanged(self):
        source = [
            (0.0, [1.0, 0.0, 0.0], [0.0, 0.0, 0.1]),
            (0.5, [1.0, 0.0, 0.0], [0.0, 0.0, 0.1]),
        ]
        obs_source = [
            (0.5, Pose3((0.5, 0.0, 0.0), ID_QUAT)),
        ]
        imu_consumed = []
        obs_consumed = []

        def imu_generator():
            for sample in source:
                imu_consumed.append(sample)
                yield sample

        def obs_generator():
            for observation in obs_source:
                obs_consumed.append(observation)
                yield observation

        sample_snapshot = [
            (sample[0], list(sample[1]), list(sample[2])) for sample in source
        ]
        result = fuse_imu_pose(imu_generator(), obs_generator(),
                               Pose3.identity(), ZERO)
        self.assertEqual(len(imu_consumed), 2)
        self.assertEqual(len(obs_consumed), 1)
        self.assertEqual(len(result), 2)
        self.assertTrue(result[1].observation_used)
        self.assertEqual(
            [(sample[0], list(sample[1]), list(sample[2])) for sample in source],
            sample_snapshot,
        )

    def test_result_is_immutable(self):
        result = fuse_imu_pose(
            [(0.0, ZERO, ZERO), (1.0, ZERO, ZERO)], [],
            Pose3.identity(), ZERO,
        )
        with self.assertRaises(AttributeError):
            result[0].timestamp = 5.0
        with self.assertRaises(TypeError):
            result[0].velocity[0] = 9.0
        self.assertIsInstance(result, tuple)
        self.assertIsInstance(result[0].velocity, tuple)
        self.assertIsInstance(result[0], FusionState)

    def test_deterministic_across_calls(self):
        samples = [
            (0.1 * i, (0.3 * i - 1.0, 0.5, -0.2 * i), (0.01, 0.02 * i, -0.01))
            for i in range(8)
        ]
        observations = [
            (0.1 * 3, Pose3((0.1, 0.0, 0.0), rotation_z(0.05))),
            (0.1 * 6, Pose3((0.4, 0.1, 0.0), ID_QUAT)),
        ]
        kwargs = dict(
            accelerometer_bias=(0.01, 0.0, 0.02),
            gyroscope_bias=(0.0, 0.001, 0.0),
            gravity=GRAVITY,
            translation_gain=0.7,
            rotation_gain=0.8,
            residual_threshold=2.0,
        )
        first = fuse_imu_pose(samples, observations, Pose3.identity(),
                              (0.1, 0.0, 0.0), **kwargs)
        second = fuse_imu_pose(samples, observations, Pose3.identity(),
                               (0.1, 0.0, 0.0), **kwargs)
        self.assertEqual(
            [(s.timestamp, s.pose, s.velocity, s.observation_used) for s in first],
            [(s.timestamp, s.pose, s.velocity, s.observation_used) for s in second],
        )

    def test_correction_propagates_to_later_states(self):
        samples = stationary_samples(3)
        observed = Pose3((2.0, 0.0, 0.0), ID_QUAT)
        result = fuse_imu_pose(
            samples, [(1.0, observed)], Pose3.identity(), ZERO,
            translation_gain=0.5,
        )
        self.assertTrue(close(result[1].pose.translation, (1.0, 0.0, 0.0)))
        self.assertTrue(close(result[2].pose.translation, (1.0, 0.0, 0.0)))


if __name__ == "__main__":
    unittest.main()
