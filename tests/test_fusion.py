import math
import unittest

from mappilot.fusion import FusionState, fuse_imu_pose
from mappilot.geometry import Pose3

ZERO = (0.0, 0.0, 0.0)
ID_QUAT = (1.0, 0.0, 0.0, 0.0)


def close(actual, expected, tol=1e-9):
    return all(abs(a - e) <= tol for a, e in zip(actual, expected))


def rotation_z(angle):
    return (math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0))


def still_samples(t0=0.0, dt=1.0, count=3):
    return [
        (t0 + dt * i, ZERO, ZERO)
        for i in range(count)
    ]


class PropagationTest(unittest.TestCase):
    def test_zero_inputs_keep_initial_state(self):
        states = fuse_imu_pose(
            still_samples(), [], Pose3.identity(), ZERO
        )
        self.assertEqual(len(states), 3)
        for index, state in enumerate(states):
            self.assertIsInstance(state, FusionState)
            self.assertEqual(state.timestamp, float(index))
            self.assertEqual(state.pose, Pose3.identity())
            self.assertEqual(state.velocity, ZERO)
            self.assertIsNone(state.observation_used)

    def test_constant_world_acceleration(self):
        samples = [
            (0.0, (2.0, 0.0, 0.0), ZERO),
            (1.0, (2.0, 0.0, 0.0), ZERO),
            (2.0, (2.0, 0.0, 0.0), ZERO),
        ]
        states = fuse_imu_pose(
            samples,
            [],
            Pose3((1.0, 0.0, 0.0), ID_QUAT),
            (0.0, 1.0, 0.0),
        )
        self.assertTrue(close(states[1].pose.translation, (2.0, 1.0, 0.0)))
        self.assertTrue(close(states[1].velocity, (2.0, 1.0, 0.0)))
        self.assertTrue(close(states[2].pose.translation, (5.0, 2.0, 0.0)))
        self.assertTrue(close(states[2].velocity, (4.0, 1.0, 0.0)))
        for state in states:
            self.assertTrue(close(state.pose.quaternion, ID_QUAT))

    def test_gravity_added_in_world_frame(self):
        samples = [(0.0, ZERO, ZERO), (1.0, ZERO, ZERO)]
        states = fuse_imu_pose(
            samples, [], Pose3.identity(), ZERO, gravity=(0.0, 0.0, -9.8)
        )
        self.assertTrue(close(states[1].pose.translation, (0.0, 0.0, -4.9)))
        self.assertTrue(close(states[1].velocity, (0.0, 0.0, -9.8)))

    def test_constant_angular_rate_right_composes(self):
        omega = (0.0, 0.0, math.pi / 2.0)
        samples = [(0.0, ZERO, omega), (1.0, ZERO, omega)]
        states = fuse_imu_pose(samples, [], Pose3.identity(), ZERO)
        self.assertTrue(close(states[1].pose.quaternion, rotation_z(math.pi / 2.0)))
        self.assertTrue(close(states[1].pose.translation, ZERO))
        self.assertTrue(close(states[1].velocity, ZERO))

    def test_specific_force_rotated_by_interval_start_rotation(self):
        samples = [
            (0.0, (0.0, 1.0, 0.0), (0.0, 0.0, math.pi / 2.0)),
            (1.0, (0.0, 1.0, 0.0), ZERO),
        ]
        states = fuse_imu_pose(samples, [], Pose3.identity(), ZERO)
        # Left-endpoint: first interval rotates accel with identity.
        self.assertTrue(close(states[1].pose.translation, (0.0, 0.5, 0.0)))
        self.assertTrue(close(states[1].velocity, (0.0, 1.0, 0.0)))
        self.assertTrue(close(states[1].pose.quaternion, rotation_z(math.pi / 2.0)))

    def test_rotation_steers_second_interval_acceleration(self):
        samples = [
            (0.0, (0.0, 1.0, 0.0), (0.0, 0.0, math.pi)),
            (1.0, (0.0, 1.0, 0.0), ZERO),
            (2.0, ZERO, ZERO),
        ]
        states = fuse_imu_pose(samples, [], Pose3.identity(), ZERO)
        # Interval 2 starts at Rz(pi): a_world = (0,-1,0).
        self.assertTrue(close(states[2].velocity, (0.0, 0.0, 0.0)))
        self.assertTrue(close(states[2].pose.translation, (0.0, 1.0, 0.0)))
        self.assertTrue(close(states[2].pose.quaternion, rotation_z(math.pi)))

    def test_biases_subtracted_from_left_sample(self):
        samples = [
            (0.0, (0.0, 1.0, 0.0), (0.0, 0.0, 0.1)),
            (1.0, (0.0, 1.0, 0.0), (0.0, 0.0, 0.1)),
        ]
        states = fuse_imu_pose(
            samples,
            [],
            Pose3.identity(),
            ZERO,
            accelerometer_bias=(0.0, 1.0, 0.0),
            gyroscope_bias=(0.0, 0.0, 0.1),
        )
        self.assertEqual(states[1].pose, Pose3.identity())
        self.assertEqual(states[1].velocity, ZERO)


class ObservationTest(unittest.TestCase):
    def test_full_gain_snaps_to_observed_pose(self):
        samples = [
            (0.0, (1.0, 0.0, 0.0), ZERO),
            (1.0, (1.0, 0.0, 0.0), ZERO),
        ]
        observed = Pose3((10.0, 0.0, 0.0), ID_QUAT)
        states = fuse_imu_pose(
            samples, [(1.0, observed)], Pose3.identity(), ZERO
        )
        self.assertIsNone(states[0].observation_used)
        self.assertTrue(states[1].observation_used)
        self.assertEqual(states[1].pose, observed)
        # Velocity is never changed directly by an observation.
        self.assertEqual(states[1].velocity, (1.0, 0.0, 0.0))

    def test_first_timestamp_observation_corrects_initial_state(self):
        observed = Pose3((5.0, 0.0, 0.0), ID_QUAT)
        states = fuse_imu_pose(
            still_samples(),
            [(0.0, observed)],
            Pose3.identity(),
            ZERO,
        )
        self.assertTrue(states[0].observation_used)
        self.assertEqual(states[0].pose, observed)
        # Propagation from the corrected state onward.
        self.assertTrue(close(states[1].pose.translation, (5.0, 0.0, 0.0)))
        self.assertTrue(close(states[2].pose.translation, (5.0, 0.0, 0.0)))

    def test_partial_translation_gain(self):
        samples = [
            (0.0, ZERO, ZERO),
            (1.0, ZERO, ZERO),
        ]
        observed = Pose3((4.0, 0.0, 0.0), ID_QUAT)
        states = fuse_imu_pose(
            samples,
            [(1.0, observed)],
            Pose3.identity(),
            ZERO,
            translation_gain=0.5,
        )
        self.assertTrue(states[1].observation_used)
        self.assertTrue(close(states[1].pose.translation, (2.0, 0.0, 0.0)))

    def test_partial_rotation_gain(self):
        samples = [(0.0, ZERO, ZERO), (1.0, ZERO, ZERO)]
        observed = Pose3(ZERO, rotation_z(math.pi / 2.0))
        states = fuse_imu_pose(
            samples,
            [(1.0, observed)],
            Pose3.identity(),
            ZERO,
            rotation_gain=0.5,
        )
        self.assertTrue(close(states[1].pose.quaternion, rotation_z(math.pi / 4.0)))

    def test_correction_right_composed_onto_predicted(self):
        # Predicted pose has a rotation and translation of its own; a pure
        # rotation fix rotates around the body frame via right composition.
        samples = [(0.0, ZERO, ZERO), (1.0, ZERO, ZERO)]
        predicted = Pose3((1.0, 2.0, 0.0), rotation_z(math.pi / 2.0))
        observed = predicted.compose(Pose3(ZERO, rotation_z(math.pi / 2.0)))
        states = fuse_imu_pose(
            samples, [(1.0, observed)], predicted, ZERO
        )
        expected = predicted.compose(Pose3(ZERO, rotation_z(math.pi / 2.0)))
        self.assertTrue(close(states[1].pose.quaternion, expected.quaternion))
        self.assertTrue(close(states[1].pose.translation, expected.translation))

    def test_empty_observations_allowed(self):
        states = fuse_imu_pose(still_samples(), iter([]), Pose3.identity(), ZERO)
        self.assertEqual(len(states), 3)
        self.assertTrue(all(state.observation_used is None for state in states))


class GateTest(unittest.TestCase):
    def test_translation_residual_above_threshold_rejected(self):
        samples = [(0.0, ZERO, ZERO), (1.0, ZERO, ZERO)]
        observed = Pose3((2.0, 0.0, 0.0), ID_QUAT)
        states = fuse_imu_pose(
            samples,
            [(1.0, observed)],
            Pose3.identity(),
            ZERO,
            residual_threshold=1.0,
        )
        self.assertFalse(states[1].observation_used)
        self.assertEqual(states[1].pose, Pose3.identity())

    def test_translation_residual_equal_to_threshold_accepted(self):
        samples = [(0.0, ZERO, ZERO), (1.0, ZERO, ZERO)]
        observed = Pose3((1.0, 0.0, 0.0), ID_QUAT)
        states = fuse_imu_pose(
            samples,
            [(1.0, observed)],
            Pose3.identity(),
            ZERO,
            residual_threshold=1.0,
        )
        self.assertTrue(states[1].observation_used)
        self.assertEqual(states[1].pose, observed)

    def test_rotation_residual_above_threshold_rejected(self):
        samples = [(0.0, ZERO, ZERO), (1.0, ZERO, ZERO)]
        observed = Pose3(ZERO, rotation_z(0.5))
        states = fuse_imu_pose(
            samples,
            [(1.0, observed)],
            Pose3.identity(),
            ZERO,
            residual_threshold=0.4,
        )
        self.assertFalse(states[1].observation_used)
        self.assertEqual(states[1].pose, Pose3.identity())

    def test_rotation_residual_equal_to_threshold_accepted(self):
        samples = [(0.0, ZERO, ZERO), (1.0, ZERO, ZERO)]
        observed = Pose3(ZERO, rotation_z(0.5))
        states = fuse_imu_pose(
            samples,
            [(1.0, observed)],
            Pose3.identity(),
            ZERO,
            residual_threshold=0.5,
        )
        self.assertTrue(states[1].observation_used)

    def test_zero_threshold_rejects_any_nonzero_residual(self):
        samples = [(0.0, ZERO, ZERO), (1.0, ZERO, ZERO)]
        for observed in (
            Pose3((1e-12, 0.0, 0.0), ID_QUAT),
            Pose3(ZERO, rotation_z(1e-12)),
        ):
            states = fuse_imu_pose(
                samples,
                [(1.0, observed)],
                Pose3.identity(),
                ZERO,
                residual_threshold=0.0,
            )
            self.assertFalse(states[1].observation_used)

    def test_rejected_observation_does_not_disturb_later_states(self):
        samples = [
            (0.0, (1.0, 0.0, 0.0), ZERO),
            (1.0, (1.0, 0.0, 0.0), ZERO),
            (2.0, (1.0, 0.0, 0.0), ZERO),
        ]
        bad_observation = Pose3((100.0, 0.0, 0.0), ID_QUAT)
        states = fuse_imu_pose(
            samples,
            [(1.0, bad_observation)],
            Pose3.identity(),
            ZERO,
            residual_threshold=1.0,
        )
        flags = [state.observation_used for state in states]
        self.assertEqual(flags, [None, False, None])
        self.assertTrue(close(states[1].pose.translation, (0.5, 0.0, 0.0)))
        self.assertTrue(close(states[2].pose.translation, (2.0, 0.0, 0.0)))

    def test_first_timestamp_observation_can_be_rejected(self):
        observed = Pose3((3.0, 0.0, 0.0), ID_QUAT)
        states = fuse_imu_pose(
            still_samples(),
            [(0.0, observed)],
            Pose3.identity(),
            ZERO,
            residual_threshold=1.0,
        )
        self.assertFalse(states[0].observation_used)
        self.assertEqual(states[0].pose, Pose3.identity())


class ValidationTest(unittest.TestCase):
    SAMPLES = [(0.0, ZERO, ZERO), (1.0, ZERO, ZERO)]

    def test_too_few_samples_raises_value_error(self):
        with self.assertRaises(ValueError):
            fuse_imu_pose([], [], Pose3.identity(), ZERO)
        with self.assertRaises(ValueError):
            fuse_imu_pose([(0.0, ZERO, ZERO)], [], Pose3.identity(), ZERO)

    def test_non_strictly_increasing_imu_timestamps_raise(self):
        with self.assertRaises(ValueError):
            fuse_imu_pose(
                [(1.0, ZERO, ZERO), (1.0, ZERO, ZERO)],
                [],
                Pose3.identity(),
                ZERO,
            )
        with self.assertRaises(ValueError):
            fuse_imu_pose(
                [(2.0, ZERO, ZERO), (1.0, ZERO, ZERO)],
                [],
                Pose3.identity(),
                ZERO,
            )

    def test_non_strictly_increasing_observation_timestamps_raise(self):
        observations = [
            (1.0, Pose3.identity()),
            (1.0, Pose3.identity()),
        ]
        with self.assertRaises(ValueError):
            fuse_imu_pose(self.SAMPLES, observations, Pose3.identity(), ZERO)
        with self.assertRaises(ValueError):
            fuse_imu_pose(
                [(0.0, ZERO, ZERO), (1.0, ZERO, ZERO), (2.0, ZERO, ZERO)],
                [(2.0, Pose3.identity()), (1.0, Pose3.identity())],
                Pose3.identity(),
                ZERO,
            )

    def test_observation_timestamp_must_hit_imu_sample(self):
        with self.assertRaises(ValueError):
            fuse_imu_pose(
                self.SAMPLES,
                [(0.5, Pose3.identity())],
                Pose3.identity(),
                ZERO,
            )

    def test_non_finite_values_raise_value_error(self):
        with self.assertRaises(ValueError):
            fuse_imu_pose(
                [(float("nan"), ZERO, ZERO), (1.0, ZERO, ZERO)],
                [],
                Pose3.identity(),
                ZERO,
            )
        with self.assertRaises(ValueError):
            fuse_imu_pose(
                [(0.0, (float("inf"), 0.0, 0.0), ZERO), (1.0, ZERO, ZERO)],
                [],
                Pose3.identity(),
                ZERO,
            )
        with self.assertRaises(ValueError):
            fuse_imu_pose(
                self.SAMPLES, [], Pose3.identity(), (0.0, float("nan"), 0.0)
            )
        with self.assertRaises(ValueError):
            fuse_imu_pose(
                self.SAMPLES,
                [],
                Pose3.identity(),
                ZERO,
                gravity=(0.0, 0.0, float("inf")),
            )
        with self.assertRaises(ValueError):
            fuse_imu_pose(
                self.SAMPLES,
                [(float("nan"), Pose3.identity())],
                Pose3.identity(),
                ZERO,
            )

    def test_non_finite_propagation_raises_value_error(self):
        huge = (1e308, 0.0, 0.0)
        with self.assertRaises(ValueError):
            fuse_imu_pose(
                [(0.0, huge, ZERO), (2.0, huge, ZERO)],
                [],
                Pose3.identity(),
                ZERO,
            )

    def test_gains_outside_unit_interval_raise(self):
        for kwargs in (
            {"translation_gain": -0.01},
            {"translation_gain": 1.01},
            {"rotation_gain": -1.0},
            {"rotation_gain": 2.0},
        ):
            with self.assertRaises(ValueError):
                fuse_imu_pose(
                    self.SAMPLES, [], Pose3.identity(), ZERO, **kwargs
                )

    def test_gain_boundaries_accepted(self):
        for gain in (0.0, 1.0):
            fuse_imu_pose(
                self.SAMPLES,
                [],
                Pose3.identity(),
                ZERO,
                translation_gain=gain,
                rotation_gain=gain,
            )

    def test_non_finite_gain_raises_value_error(self):
        with self.assertRaises(ValueError):
            fuse_imu_pose(
                self.SAMPLES, [], Pose3.identity(), ZERO,
                translation_gain=float("inf"),
            )

    def test_boolean_gain_raises_type_error(self):
        with self.assertRaises(TypeError):
            fuse_imu_pose(
                self.SAMPLES, [], Pose3.identity(), ZERO,
                translation_gain=True,
            )

    def test_threshold_validation(self):
        with self.assertRaises(ValueError):
            fuse_imu_pose(
                self.SAMPLES, [], Pose3.identity(), ZERO,
                residual_threshold=-0.01,
            )
        with self.assertRaises(ValueError):
            fuse_imu_pose(
                self.SAMPLES, [], Pose3.identity(), ZERO,
                residual_threshold=float("nan"),
            )
        with self.assertRaises(ValueError):
            fuse_imu_pose(
                self.SAMPLES, [], Pose3.identity(), ZERO,
                residual_threshold=float("-inf"),
            )
        with self.assertRaises(TypeError):
            fuse_imu_pose(
                self.SAMPLES, [], Pose3.identity(), ZERO,
                residual_threshold=True,
            )
        # Zero threshold is valid.
        fuse_imu_pose(
            self.SAMPLES, [], Pose3.identity(), ZERO,
            residual_threshold=0.0,
        )

    def test_type_errors(self):
        with self.assertRaises(TypeError):
            fuse_imu_pose(42, [], Pose3.identity(), ZERO)
        with self.assertRaises(TypeError):
            fuse_imu_pose(None, [], Pose3.identity(), ZERO)
        with self.assertRaises(TypeError):
            fuse_imu_pose("not-iterable", [], Pose3.identity(), ZERO)
        with self.assertRaises(TypeError):
            fuse_imu_pose([1234, 5678], [], Pose3.identity(), ZERO)
        with self.assertRaises(TypeError):
            fuse_imu_pose(
                [(0.0, ZERO), (1.0, ZERO, ZERO)],
                [],
                Pose3.identity(),
                ZERO,
            )
        with self.assertRaises(TypeError):
            fuse_imu_pose(
                [(0.0, ZERO, ZERO, ZERO), (1.0, ZERO, ZERO)],
                [],
                Pose3.identity(),
                ZERO,
            )
        with self.assertRaises(TypeError):
            fuse_imu_pose(
                [(0.0, (0.0, 0.0), ZERO), (1.0, ZERO, ZERO)],
                [],
                Pose3.identity(),
                ZERO,
            )
        with self.assertRaises(TypeError):
            fuse_imu_pose(
                [(True, ZERO, ZERO), (1.0, ZERO, ZERO)],
                [],
                Pose3.identity(),
                ZERO,
            )
        with self.assertRaises(TypeError):
            fuse_imu_pose(
                [(0.0, (True, 0.0, 0.0), ZERO), (1.0, ZERO, ZERO)],
                [],
                Pose3.identity(),
                ZERO,
            )
        with self.assertRaises(TypeError):
            fuse_imu_pose(
                self.SAMPLES, [], Pose3.identity(), (0.0, 0.0)
            )
        with self.assertRaises(TypeError):
            fuse_imu_pose(
                self.SAMPLES, [], Pose3.identity(), ZERO,
                accelerometer_bias=1.5,
            )
        with self.assertRaises(TypeError):
            fuse_imu_pose(
                self.SAMPLES,
                [(0.0, ZERO)],
                Pose3.identity(),
                ZERO,
            )
        with self.assertRaises(TypeError):
            fuse_imu_pose(
                self.SAMPLES,
                [(0.0, (0.0, 0.0))],
                Pose3.identity(),
                ZERO,
            )
        with self.assertRaises(TypeError):
            fuse_imu_pose(
                self.SAMPLES,
                [(0.0, "pose")],
                Pose3.identity(),
                ZERO,
            )

    def test_initial_pose_wrong_type_raises(self):
        with self.assertRaises(TypeError):
            fuse_imu_pose(self.SAMPLES, [], object(), ZERO)
        with self.assertRaises(TypeError):
            fuse_imu_pose(self.SAMPLES, [], ((0.0, 0.0, 0.0), ID_QUAT), ZERO)


class ContractTest(unittest.TestCase):
    def test_generators_consumed_once_and_inputs_unchanged(self):
        source = [
            (0.0, [1.0, 0.0, 0.0], [0.0, 0.0, 0.1]),
            (0.5, [1.0, 0.0, 0.0], [0.0, 0.0, 0.1]),
            (1.0, [1.0, 0.0, 0.0], [0.0, 0.0, 0.1]),
        ]
        obs_source = [
            (0.5, Pose3((0.1, 0.0, 0.0), ID_QUAT)),
        ]
        consumed_samples = []
        consumed_observations = []

        def sample_gen():
            for sample in source:
                consumed_samples.append(sample)
                yield sample

        def observation_gen():
            for observation in obs_source:
                consumed_observations.append(observation)
                yield observation

        snapshot = [
            (sample[0], list(sample[1]), list(sample[2])) for sample in source
        ]
        fuse_imu_pose(sample_gen(), observation_gen(), Pose3.identity(), ZERO)
        self.assertEqual(len(consumed_samples), 3)
        self.assertEqual(len(consumed_observations), 1)
        self.assertEqual(
            [(sample[0], list(sample[1]), list(sample[2])) for sample in source],
            snapshot,
        )

    def test_result_is_immutable(self):
        states = fuse_imu_pose(
            [(0.0, ZERO, ZERO), (1.0, ZERO, ZERO)],
            [],
            Pose3.identity(),
            ZERO,
        )
        with self.assertRaises(AttributeError):
            states[0].timestamp = 5.0
        with self.assertRaises(TypeError):
            states[0].velocity[0] = 9.0
        self.assertIsInstance(states, tuple)
        self.assertIsInstance(states[0].velocity, tuple)

    def test_deterministic_across_calls(self):
        samples = [
            (0.1 * i, (0.3 * i - 1.0, 0.5, -0.2 * i), (0.01, 0.02 * i, -0.01))
            for i in range(6)
        ]
        observations = [
            (0.2, Pose3((0.1, 0.0, 0.0), rotation_z(0.05))),
            (0.4, Pose3((0.3, 0.0, 0.0), rotation_z(0.1))),
        ]
        kwargs = dict(
            accelerometer_bias=(0.01, 0.0, 0.02),
            gyroscope_bias=(0.0, 0.001, 0.0),
            gravity=(0.0, 0.0, -9.81),
            translation_gain=0.7,
            rotation_gain=0.8,
            residual_threshold=5.0,
        )
        first = fuse_imu_pose(
            samples, observations, Pose3.identity(), (0.1, 0.0, 0.0), **kwargs
        )
        second = fuse_imu_pose(
            samples, observations, Pose3.identity(), (0.1, 0.0, 0.0), **kwargs
        )
        self.assertEqual(len(first), len(second))
        for a, b in zip(first, second):
            self.assertEqual(a.timestamp, b.timestamp)
            self.assertEqual(a.pose, b.pose)
            self.assertEqual(a.velocity, b.velocity)
            self.assertEqual(a.observation_used, b.observation_used)

    def test_flags_distinguish_absent_accepted_rejected(self):
        samples = [
            (0.0, ZERO, ZERO),
            (1.0, ZERO, ZERO),
            (2.0, ZERO, ZERO),
        ]
        observations = [
            (0.0, Pose3((0.5, 0.0, 0.0), ID_QUAT)),
            (1.0, Pose3((2.0, 0.0, 0.0), ID_QUAT)),
        ]
        states = fuse_imu_pose(
            samples,
            observations,
            Pose3.identity(),
            ZERO,
            residual_threshold=1.0,
        )
        self.assertEqual(
            [state.observation_used for state in states],
            [True, False, None],
        )


if __name__ == "__main__":
    unittest.main()
