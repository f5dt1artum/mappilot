import importlib
import inspect
import math
import sys
import unittest

from mappilot.geometry import Pose3
from mappilot.calibration import (
    ExtrinsicCalibrationResult,
    calibrate_extrinsic,
)


def rotation_z(angle):
    return Pose3(
        (0.0, 0.0, 0.0),
        (math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0)),
    )


def pose_close(test_case, actual, expected, tol=1e-9):
    for a, e in zip(actual.translation, expected.translation):
        test_case.assertAlmostEqual(a, e, delta=tol)
    qa, qe = actual.quaternion, expected.quaternion
    # q and -q describe the same rotation.
    test_case.assertTrue(
        min(
            max(abs(a - e) for a, e in zip(qa, qe)),
            max(abs(a + e) for a, e in zip(qa, qe)),
        )
        <= tol
    )


def residual(extrinsic, body_motion, sensor_motion):
    return body_motion.compose(extrinsic).inverse().compose(
        extrinsic.compose(sensor_motion)
    ).log()


def objective(observations, extrinsic):
    return sum(
        weight * sum(component * component for component in residual(extrinsic, b, s))
        for b, s, weight in observations
    )


class CalibrateExtrinsicTest(unittest.TestCase):
    def setUp(self):
        self.true_extrinsic = Pose3.exp((0.15, -0.2, 0.1, 0.5, -0.3, 0.2))
        # Body motions span at least two distinct rotation axes so the
        # extrinsic is fully observable.
        self.body_motions = (
            Pose3.exp((0.3, 0.1, 0.05, 1.0, 0.2, -0.3)),
            Pose3.exp((-0.1, 0.25, 0.1, -0.5, 0.4, 0.1)),
            Pose3.exp((0.05, -0.1, 0.3, 0.2, -0.7, 0.2)),
            Pose3.exp((0.2, 0.2, -0.15, 0.3, 0.5, 0.5)),
        )
        self.observations = self._consistent(self.body_motions)

    def _consistent(self, body_motions, extrinsic=None, weights=1.0):
        extrinsic = self.true_extrinsic if extrinsic is None else extrinsic
        records = []
        for index, body in enumerate(body_motions):
            sensor = extrinsic.inverse().compose(body).compose(extrinsic)
            weight = weights[index] if isinstance(weights, (list, tuple)) else weights
            records.append((body, sensor, weight))
        return records

    def _perturbed(self, scale=0.03):
        # Deterministic, non-random perturbations of the sensor motions.
        records = []
        for index, body in enumerate(self.body_motions):
            sensor = self.true_extrinsic.inverse().compose(body).compose(
                self.true_extrinsic
            )
            delta = tuple(
                scale * math.sin(1.7 * index + 0.9 * k) for k in range(6)
            )
            records.append((body, Pose3.exp(delta).compose(sensor), 1.0))
        return records

    # -- recovery ------------------------------------------------------------

    def test_recovers_extrinsic_from_identity(self):
        result = calibrate_extrinsic(self.observations, tolerance=1e-12)
        self.assertIsInstance(result, ExtrinsicCalibrationResult)
        self.assertTrue(result.converged)
        self.assertLess(result.final_error, 1e-20)
        self.assertLessEqual(result.final_error, result.initial_error)
        pose_close(self, result.sensor_to_body, self.true_extrinsic, tol=1e-9)

    def test_recovers_from_non_identity_initial_guess(self):
        initial = Pose3.exp((0.4, -0.35, 0.25, 0.9, -0.7, 0.55))
        result = calibrate_extrinsic(self.observations, initial_pose=initial,
                                     tolerance=1e-12)
        self.assertTrue(result.converged)
        self.assertLess(result.final_error, 1e-18)
        pose_close(self, result.sensor_to_body, self.true_extrinsic, tol=1e-8)

    def test_recovers_with_three_observations(self):
        result = calibrate_extrinsic(self.observations[:3], tolerance=1e-12)
        self.assertTrue(result.converged)
        self.assertLess(result.final_error, 1e-18)
        pose_close(self, result.sensor_to_body, self.true_extrinsic, tol=1e-8)

    def test_pure_rotation_motions_observe_translation_too(self):
        # Rotations about two non-parallel axes constrain all six DoF even
        # when every motion is a pure rotation.
        bodies = (
            Pose3.exp((0.4, 0.0, 0.0, 0.0, 0.0, 0.0)),
            Pose3.exp((0.0, 0.35, 0.0, 0.0, 0.0, 0.0)),
            Pose3.exp((0.2, 0.2, 0.0, 0.0, 0.0, 0.0)),
        )
        observations = self._consistent(bodies)
        result = calibrate_extrinsic(
            observations,
            initial_pose=Pose3.exp((0.3, 0.3, 0.3, 0.0, 0.0, 0.0)),
            tolerance=1e-12,
        )
        self.assertTrue(result.converged)
        self.assertLess(result.final_error, 1e-20)
        pose_close(self, result.sensor_to_body, self.true_extrinsic, tol=1e-8)

    def test_angles_near_pi(self):
        bodies = (
            Pose3.exp((2.9, 0.1, 0.05, 1.0, 0.2, -0.3)),
            Pose3.exp((-0.1, 2.7, 0.1, -0.5, 0.4, 0.1)),
            Pose3.exp((0.05, -0.1, 3.0, 0.2, -0.7, 0.2)),
            Pose3.exp((0.2, 0.2, -2.8, 0.3, 0.5, 0.5)),
        )
        observations = self._consistent(bodies, weights=2.0)
        result = calibrate_extrinsic(
            observations,
            initial_pose=Pose3.exp((0.5, -0.4, 0.3, 1.0, -0.8, 0.6)),
            tolerance=1e-11,
        )
        self.assertTrue(result.converged)
        self.assertLess(result.final_error, 1e-20)
        pose_close(self, result.sensor_to_body, self.true_extrinsic, tol=1e-8)

    # -- result consistency --------------------------------------------------

    def test_residuals_in_input_order_and_recomputable(self):
        result = calibrate_extrinsic(
            self._perturbed(),
            initial_pose=Pose3.exp((0.2, 0.0, 0.0, 0.0, 0.0, 0.0)),
            tolerance=1e-13,
        )
        self.assertEqual(len(result.residuals), len(self.observations))
        for record, reported in zip(self._perturbed(), result.residuals):
            body, sensor, _ = record
            expected = residual(result.sensor_to_body, body, sensor)
            self.assertEqual(len(reported), 6)
            for actual, wanted in zip(reported, expected):
                self.assertAlmostEqual(actual, wanted, delta=1e-12)

    def test_errors_recomputed_from_poses(self):
        observations = self._perturbed()
        initial = Pose3.exp((-0.3, 0.25, -0.1, 0.4, -0.2, 0.35))
        result = calibrate_extrinsic(observations, initial_pose=initial,
                                     tolerance=1e-13)
        self.assertAlmostEqual(
            result.initial_error, objective(observations, initial), places=12
        )
        self.assertAlmostEqual(
            result.final_error,
            objective(observations, result.sensor_to_body),
            places=12,
        )
        self.assertAlmostEqual(
            result.final_error,
            sum(
                weight * sum(c * c for c in r)
                for (_, _, weight), r in zip(observations, result.residuals)
            ),
            places=14,
        )

    def test_result_is_least_squares_optimum(self):
        observations = self._perturbed()
        result = calibrate_extrinsic(observations, tolerance=1e-13)
        # No other reachable estimate can beat the weighted optimum; in
        # particular the generating extrinsic is no better than the result.
        self.assertLessEqual(
            result.final_error, objective(observations, self.true_extrinsic) + 1e-14
        )

    def test_weights_change_weighted_optimum(self):
        observations = self._perturbed()
        uniform = calibrate_extrinsic(observations, tolerance=1e-13)
        weighted_records = [
            (b, s, 100.0 if index == 0 else 1.0)
            for index, (b, s, _) in enumerate(observations)
        ]
        weighted = calibrate_extrinsic(weighted_records, tolerance=1e-13)
        self.assertTrue(uniform.converged and weighted.converged)
        self.assertNotEqual(
            weighted.sensor_to_body, uniform.sensor_to_body
        )
        self.assertLessEqual(
            weighted.final_error,
            objective(weighted_records, uniform.sensor_to_body) + 1e-14,
        )

    # -- iteration semantics -------------------------------------------------

    def test_zero_initial_error_short_circuits(self):
        # Starting at the generating extrinsic makes every residual zero, so
        # no update is needed.
        result = calibrate_extrinsic(
            self.observations, initial_pose=self.true_extrinsic, tolerance=1e-12
        )
        self.assertTrue(result.converged)
        self.assertEqual(result.iterations, 0)
        self.assertLess(result.initial_error, 1e-25)
        self.assertLess(result.final_error, 1e-25)
        self.assertEqual(result.sensor_to_body, self.true_extrinsic)

    def test_identity_extrinsic_zero_error_short_circuits(self):
        # Sensor motions equal to body motions describe a unit extrinsic, so
        # the default identity start is already exact.
        observations = [(b, b, 1.0) for b in self.body_motions]
        result = calibrate_extrinsic(observations, tolerance=1e-12)
        self.assertTrue(result.converged)
        self.assertEqual(result.iterations, 0)
        self.assertLess(result.final_error, 1e-25)
        self.assertEqual(result.sensor_to_body, Pose3.identity())

    def test_initial_error_within_tolerance_short_circuits(self):
        initial = Pose3.exp((0.001, 0.0, 0.0, 0.001, 0.0, 0.0))
        result = calibrate_extrinsic(self.observations, initial_pose=initial,
                                     tolerance=1.0)
        self.assertTrue(result.converged)
        self.assertEqual(result.iterations, 0)
        self.assertIs(result.sensor_to_body, initial)
        self.assertGreater(result.initial_error, 0.0)
        self.assertEqual(result.initial_error, result.final_error)

    def test_max_iterations_exhaustion_not_converged(self):
        result = calibrate_extrinsic(
            self.observations,
            initial_pose=Pose3.exp((0.5, -0.4, 0.3, 1.0, -0.8, 0.6)),
            max_iterations=1,
            tolerance=1e-15,
        )
        self.assertFalse(result.converged)
        self.assertEqual(result.iterations, 1)
        self.assertLess(result.final_error, result.initial_error)
        self.assertAlmostEqual(
            result.final_error,
            objective(self.observations, result.sensor_to_body),
            places=12,
        )

    def test_convergence_on_plateau(self):
        # Records with an identity body motion carry a residual but zero
        # Jacobian rows (the residual log(X^-1 X S) = log(S) is independent
        # of X), so they add objective without adding gradient; the pure
        # rotations about two distinct axes keep the extrinsic fully
        # observable. The first update is then exactly zero, the error change
        # is zero, and calibration converges in one iteration even though a
        # residual remains.
        offset = Pose3.exp((0.1, 0.0, 0.0, 0.0, 0.0, 0.0))
        conflicting = Pose3.exp((0.0, 0.0, 0.0, -0.1, 0.0, 0.0))
        anchor_y = Pose3.exp((0.0, 0.3, 0.0, 0.0, 0.0, 0.0))
        anchor_z = Pose3.exp((0.0, 0.0, 0.35, 0.0, 0.0, 0.0))
        observations = [
            (Pose3.identity(), offset, 1.0),
            (Pose3.identity(), conflicting, 1.0),
            (anchor_y, anchor_y, 1.0),
            (anchor_z, anchor_z, 1.0),
        ]
        result = calibrate_extrinsic(observations, tolerance=1e-15)
        self.assertTrue(result.converged)
        self.assertEqual(result.iterations, 1)
        self.assertEqual(result.sensor_to_body, Pose3.identity())
        self.assertGreater(result.final_error, 0.0)
        self.assertEqual(result.initial_error, result.final_error)

    def test_iterations_bounded_by_max(self):
        for limit in (1, 2, 3):
            result = calibrate_extrinsic(
                self.observations,
                initial_pose=Pose3.exp((0.5, -0.4, 0.3, 1.0, -0.8, 0.6)),
                max_iterations=limit,
                tolerance=1e-18,
            )
            self.assertLessEqual(result.iterations, limit)
            self.assertLessEqual(result.final_error, result.initial_error)

    # -- determinism / input handling / immutability -------------------------

    def test_deterministic_across_calls(self):
        first = calibrate_extrinsic(self._perturbed(), tolerance=1e-12)
        second = calibrate_extrinsic(self._perturbed(), tolerance=1e-12)
        self.assertEqual(first.sensor_to_body, second.sensor_to_body)
        self.assertEqual(first.converged, second.converged)
        self.assertEqual(first.iterations, second.iterations)
        self.assertEqual(first.initial_error, second.initial_error)
        self.assertEqual(first.final_error, second.final_error)
        self.assertEqual(first.residuals, second.residuals)

    def test_generator_consumed_exactly_once(self):
        records = list(self.observations)

        def generator():
            yield from records

        source = generator()
        result = calibrate_extrinsic(source)
        self.assertTrue(result.converged)
        with self.assertRaises(StopIteration):
            next(source)

    def test_caller_records_not_modified(self):
        records = [list(record) for record in self.observations]
        calibrate_extrinsic(records, tolerance=1e-12)
        self.assertEqual(len(records), len(self.observations))
        for before, after in zip(self.observations, records):
            self.assertEqual(len(after), 3)
            self.assertIs(after[0], before[0])
            self.assertIs(after[1], before[1])
            self.assertEqual(after[2], before[2])

    def test_result_is_immutable(self):
        result = calibrate_extrinsic(self.observations)
        with self.assertRaises(AttributeError):
            result.converged = False  # type: ignore[misc]
        with self.assertRaises(TypeError):
            result.residuals[0] = None  # type: ignore[index]

    def test_duplicate_observations_accepted(self):
        # Repeated motions are still well-formed as long as the set remains
        # observable.
        observations = [self.observations[0], self.observations[0],
                        self.observations[1], self.observations[2]]
        result = calibrate_extrinsic(observations, tolerance=1e-12)
        self.assertTrue(result.converged)
        self.assertEqual(len(result.residuals), 4)
        pose_close(self, result.sensor_to_body, self.true_extrinsic, tol=1e-8)

    def test_import_does_not_start_http_service(self):
        sys.modules.pop("mappilot.calibration", None)
        sys.modules.pop("http.server", None)
        importlib.import_module("mappilot.calibration")
        self.assertNotIn("http.server", sys.modules)

    def test_defaults_are_50_and_1e_6(self):
        signature = inspect.signature(calibrate_extrinsic)
        self.assertIsNone(signature.parameters["initial_pose"].default)
        self.assertEqual(signature.parameters["max_iterations"].default, 50)
        self.assertEqual(signature.parameters["tolerance"].default, 1e-6)


class CalibrateExtrinsicValidationTest(unittest.TestCase):
    def setUp(self):
        self.extrinsic = Pose3.exp((0.1, 0.2, 0.3, 0.5, -0.3, 0.2))
        self.body_motions = (
            Pose3.exp((0.3, 0.1, 0.05, 1.0, 0.2, -0.3)),
            Pose3.exp((-0.1, 0.25, 0.1, -0.5, 0.4, 0.1)),
            Pose3.exp((0.05, -0.1, 0.3, 0.2, -0.7, 0.2)),
        )
        self.observations = [
            (b, self.extrinsic.inverse().compose(b).compose(self.extrinsic), 1.0)
            for b in self.body_motions
        ]

    def assert_raises(self, exception, observations=None, **kwargs):
        if observations is None:
            observations = self.observations
        with self.assertRaises(exception):
            calibrate_extrinsic(observations, **kwargs)

    # -- type errors ---------------------------------------------------------

    def test_non_iterable_observations_type_error(self):
        for bad in (42, None, True):
            with self.assertRaises(TypeError):
                calibrate_extrinsic(bad)
        self.assert_raises(TypeError, observations="abc")

    def test_malformed_record_type_error(self):
        self.assert_raises(
            TypeError,
            observations=[(Pose3.identity(), 1.0)] * 3,
        )
        self.assert_raises(
            TypeError,
            observations=[(Pose3.identity(), Pose3.identity(), 1.0, 9)] * 3,
        )
        self.assert_raises(TypeError, observations=[7] * 3)
        self.assert_raises(
            TypeError,
            observations=[("body", "sensor", 1.0)] * 3,
        )

    def test_motion_wrong_type_error(self):
        good = Pose3.identity()
        self.assert_raises(
            TypeError,
            observations=[("pose", good, 1.0), *[
                (b, s, w) for b, s, w in self.observations[:2]
            ]],
        )
        self.assert_raises(
            TypeError,
            observations=[(good, "pose", 1.0), *[
                (b, s, w) for b, s, w in self.observations[:2]
            ]],
        )

    def test_initial_pose_wrong_type_error(self):
        self.assert_raises(TypeError, initial_pose="pose")
        self.assert_raises(TypeError, initial_pose=1)
        self.assert_raises(TypeError, initial_pose=(0.0, 0.0, 0.0))

    def test_weight_wrong_type_error(self):
        for bad in (True, "1.0", 1 + 2j):
            observations = [
                (b, s, bad if index == 2 else 1.0)
                for index, (b, s, _) in enumerate(self.observations)
            ]
            self.assert_raises(TypeError, observations=observations)

    def test_parameter_type_errors(self):
        self.assert_raises(TypeError, max_iterations=1.5)
        self.assert_raises(TypeError, max_iterations=True)
        self.assert_raises(TypeError, max_iterations="50")
        self.assert_raises(TypeError, tolerance="1e-6")
        self.assert_raises(TypeError, tolerance=True)

    # -- value errors --------------------------------------------------------

    def test_fewer_than_three_observations_value_error(self):
        for count in (0, 1, 2):
            self.assert_raises(
                ValueError, observations=self.observations[:count]
            )

    def test_weight_values_value_error(self):
        for bad in (0.0, -0.5):
            observations = [
                (b, s, bad if index == 2 else 1.0)
                for index, (b, s, _) in enumerate(self.observations)
            ]
            self.assert_raises(ValueError, observations=observations)
        for bad in (float("nan"), float("inf"), float("-inf")):
            observations = [
                (b, s, bad if index == 2 else 1.0)
                for index, (b, s, _) in enumerate(self.observations)
            ]
            self.assert_raises(ValueError, observations=observations)

    def test_parameter_values_value_error(self):
        self.assert_raises(ValueError, max_iterations=0)
        self.assert_raises(ValueError, max_iterations=-5)
        self.assert_raises(ValueError, tolerance=0.0)
        self.assert_raises(ValueError, tolerance=-1e-6)
        self.assert_raises(ValueError, tolerance=float("inf"))
        self.assert_raises(ValueError, tolerance=float("nan"))

    def test_all_identity_motions_degenerate(self):
        observations = [
            (Pose3.identity(), Pose3.identity(), 1.0) for _ in range(4)
        ]
        self.assert_raises(ValueError, observations=observations)

    def test_translation_only_motions_degenerate(self):
        # Pure translations span 3D at most, leaving rotation unobservable.
        bodies = (Pose3((1.0, 0.0, 0.0)),
                  Pose3((0.0, 1.0, 0.0)),
                  Pose3((0.0, 0.0, 1.0)))
        observations = [
            (b, self.extrinsic.inverse().compose(b).compose(self.extrinsic), 1.0)
            for b in bodies
        ]
        self.assert_raises(ValueError, observations=observations)

    def test_single_rotation_axis_degenerate(self):
        bodies = (
            Pose3.exp((0.3, 0.0, 0.0, 0.0, 0.0, 0.0)),
            Pose3.exp((0.15, 0.0, 0.0, 0.0, 0.0, 0.0)),
            Pose3.exp((0.4, 0.0, 0.0, 0.0, 0.0, 0.0)),
        )
        observations = [
            (b, self.extrinsic.inverse().compose(b).compose(self.extrinsic), 1.0)
            for b in bodies
        ]
        self.assert_raises(ValueError, observations=observations)

    def test_single_rotation_axis_translations_cannot_help(self):
        # Rotations all about x, even with translations spanning y and z,
        # leaves the extrinsic under-constrained.
        bodies = (
            Pose3.exp((0.3, 0.0, 0.0, 1.0, 0.0, 0.0)),
            Pose3.exp((0.1, 0.0, 0.0, 0.0, 1.0, 0.0)),
            Pose3.exp((0.2, 0.0, 0.0, 0.0, 0.0, 1.0)),
        )
        observations = [
            (b, self.extrinsic.inverse().compose(b).compose(self.extrinsic), 1.0)
            for b in bodies
        ]
        self.assert_raises(ValueError, observations=observations)

    def test_collinear_rotation_vectors_degenerate(self):
        # Rotation vectors that are scalar multiples of one another are a
        # single effective axis.
        bodies = (
            Pose3.exp((0.3, 0.1, 0.05, 1.0, 0.2, -0.3)),
            Pose3.exp((0.6, 0.2, 0.1, -0.4, 0.3, 0.2)),
            Pose3.exp((-0.3, -0.1, -0.05, 0.2, 0.1, 0.5)),
        )
        observations = [
            (b, self.extrinsic.inverse().compose(b).compose(self.extrinsic), 1.0)
            for b in bodies
        ]
        self.assert_raises(ValueError, observations=observations)

    def test_degeneracy_independent_of_initial_pose(self):
        bodies = (
            Pose3.exp((0.3, 0.0, 0.0, 0.0, 0.5, 0.2)),
            Pose3.exp((0.1, 0.0, 0.0, 0.0, 0.1, 0.4)),
            Pose3.exp((0.4, 0.0, 0.0, 0.0, -0.3, 0.1)),
        )
        observations = [
            (b, self.extrinsic.inverse().compose(b).compose(self.extrinsic), 1.0)
            for b in bodies
        ]
        self.assert_raises(
            ValueError,
            observations=observations,
            initial_pose=Pose3.exp((0.1, 0.2, 0.3, 0.4, 0.5, 0.6)),
        )

    def test_inconsistent_motions_still_calibrate(self):
        # Degeneracy is about observability, not about whether the motions
        # agree on one extrinsic: noisy, inconsistent motions must be fit by
        # least squares rather than rejected.
        observations = []
        for index, body in enumerate(self.body_motions + (
            Pose3.exp((0.2, 0.2, -0.15, 0.3, 0.5, 0.5)),
        )):
            sensor = self.extrinsic.inverse().compose(body).compose(self.extrinsic)
            delta = tuple(0.01 * math.sin(index + k) for k in range(6))
            observations.append((body, Pose3.exp(delta).compose(sensor), 1.0))
        result = calibrate_extrinsic(observations, tolerance=1e-12)
        self.assertTrue(result.converged)
        self.assertGreater(result.final_error, 0.0)


if __name__ == "__main__":
    unittest.main()
