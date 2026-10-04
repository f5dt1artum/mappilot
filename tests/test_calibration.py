import math
import subprocess
import sys
import unittest

from mappilot.calibration import ExtrinsicCalibrationResult, calibrate_extrinsic
from mappilot.geometry import Pose3


def make_motions():
    """Body motions whose rotation axes span three directions."""
    return [
        Pose3.exp((0.3, 0.1, -0.2, 0.5, -0.3, 0.2)),
        Pose3.exp((-0.2, 0.4, 0.1, -0.1, 0.6, 0.3)),
        Pose3.exp((0.1, -0.3, 0.5, 0.2, 0.1, -0.4)),
        Pose3.exp((0.4, 0.2, 0.3, -0.3, 0.2, 0.5)),
    ]


TRUE_EXTRINSIC = Pose3.exp((0.2, -0.1, 0.3, 0.4, -0.5, 0.2))


def make_observations(extrinsic=TRUE_EXTRINSIC, motions=None):
    """Exact paired motions: sensor_motion = X^-1 * body_motion * X."""
    if motions is None:
        motions = make_motions()
    return [
        (
            body,
            extrinsic.inverse().compose(body).compose(extrinsic),
            1.0 + 0.1 * index,
        )
        for index, body in enumerate(motions)
    ]


def objective(observations, extrinsic):
    total = 0.0
    for body, sensor, weight in observations:
        residual = (
            body.compose(extrinsic).inverse().compose(extrinsic.compose(sensor)).log()
        )
        total += weight * sum(component * component for component in residual)
    return total


class ConvergenceTest(unittest.TestCase):
    def test_recovers_extrinsic_from_identity_start(self) -> None:
        observations = make_observations()
        result = calibrate_extrinsic(observations)
        self.assertIsInstance(result, ExtrinsicCalibrationResult)
        self.assertTrue(result.converged)
        self.assertGreater(result.iterations, 0)
        self.assertLess(result.final_error, 1e-10)
        for actual, expected in zip(
            result.sensor_to_body.translation, TRUE_EXTRINSIC.translation
        ):
            self.assertAlmostEqual(actual, expected, places=6)
        for actual, expected in zip(
            result.sensor_to_body.quaternion, TRUE_EXTRINSIC.quaternion
        ):
            self.assertAlmostEqual(actual, expected, places=6)

    def test_recovers_extrinsic_from_perturbed_start(self) -> None:
        observations = make_observations()
        start = Pose3.exp((0.05, -0.04, 0.03, 0.1, 0.1, -0.1))
        result = calibrate_extrinsic(observations, initial_pose=start)
        self.assertTrue(result.converged)
        self.assertLess(result.final_error, 1e-10)
        for actual, expected in zip(
            result.sensor_to_body.translation, TRUE_EXTRINSIC.translation
        ):
            self.assertAlmostEqual(actual, expected, places=6)

    def test_result_fields_recompute_from_poses(self) -> None:
        observations = make_observations()
        start = Pose3.exp((0.1, 0.0, -0.1, 0.2, 0.0, 0.1))
        result = calibrate_extrinsic(observations, initial_pose=start)
        self.assertAlmostEqual(
            result.initial_error, objective(observations, start), places=12
        )
        self.assertAlmostEqual(
            result.final_error,
            objective(observations, result.sensor_to_body),
            places=12,
        )
        self.assertEqual(len(result.residuals), len(observations))
        for (body, sensor, _), residual in zip(observations, result.residuals):
            expected = (
                body.compose(result.sensor_to_body)
                .inverse()
                .compose(result.sensor_to_body.compose(sensor))
                .log()
            )
            self.assertEqual(expected, residual)
        weighted = sum(
            weight * sum(component * component for component in residual)
            for (_, _, weight), residual in zip(observations, result.residuals)
        )
        self.assertAlmostEqual(result.final_error, weighted, places=12)

    def test_initial_error_within_tolerance_succeeds_immediately(self) -> None:
        observations = make_observations()
        result = calibrate_extrinsic(observations, initial_pose=TRUE_EXTRINSIC)
        self.assertTrue(result.converged)
        self.assertEqual(result.iterations, 0)
        self.assertEqual(result.sensor_to_body, TRUE_EXTRINSIC)
        self.assertEqual(result.initial_error, result.final_error)

    def test_iteration_exhaustion_returns_last_result(self) -> None:
        observations = make_observations()
        start = Pose3.exp((0.5, -0.4, 0.3, 1.0, 1.0, -1.0))
        result = calibrate_extrinsic(
            observations,
            initial_pose=start,
            max_iterations=1,
            tolerance=1e-300,
        )
        self.assertFalse(result.converged)
        self.assertEqual(result.iterations, 1)
        self.assertLess(result.final_error, result.initial_error)

    def test_accepts_generator_and_consumes_it_once(self) -> None:
        consumed = []

        def source():
            for record in make_observations():
                consumed.append(record)
                yield record

        result = calibrate_extrinsic(source())
        self.assertTrue(result.converged)
        self.assertEqual(len(consumed), 4)

    def test_deterministic_for_identical_input(self) -> None:
        first = calibrate_extrinsic(make_observations())
        second = calibrate_extrinsic(make_observations())
        self.assertEqual(first, second)

    def test_does_not_mutate_caller_objects(self) -> None:
        observations = make_observations()
        snapshot = list(observations)
        calibrate_extrinsic(observations)
        self.assertEqual(observations, snapshot)

    def test_repeated_motions_are_valid(self) -> None:
        motions = make_motions()[:3]
        observations = make_observations(motions=motions * 2)
        result = calibrate_extrinsic(observations)
        self.assertTrue(result.converged)
        self.assertLess(result.final_error, 1e-10)


class DegeneracyTest(unittest.TestCase):
    def test_all_identity_motions_are_degenerate(self) -> None:
        observations = [(Pose3.identity(), Pose3.identity(), 1.0) for _ in range(4)]
        with self.assertRaises(ValueError):
            calibrate_extrinsic(observations)

    def test_zero_rotations_are_degenerate(self) -> None:
        observations = [
            (Pose3((1.0, 0.0, 0.0)), Pose3((0.5, 0.0, 0.0)), 1.0),
            (Pose3((0.0, 1.0, 0.0)), Pose3((0.0, 0.5, 0.0)), 1.0),
            (Pose3((0.0, 0.0, 1.0)), Pose3((0.0, 0.0, 0.5)), 1.0),
        ]
        with self.assertRaises(ValueError):
            calibrate_extrinsic(observations)

    def test_single_rotation_axis_is_degenerate(self) -> None:
        motions = [
            Pose3.exp((0.0, 0.0, 0.4, 0.5, 0.2, 0.1)),
            Pose3.exp((0.0, 0.0, -0.3, -0.2, 0.6, 0.0)),
            Pose3.exp((0.0, 0.0, 0.2, 0.1, -0.4, 0.3)),
        ]
        with self.assertRaises(ValueError):
            calibrate_extrinsic(make_observations(motions=motions))

    def test_single_axis_with_translation_coupling_still_degenerate(self) -> None:
        # Translations not parallel to the axis can pin the rotation about
        # it, but never the translation along it.
        motions = [
            Pose3.exp((0.0, 0.0, 0.4, 0.5, 0.2, 0.1)),
            Pose3.exp((0.0, 0.0, -0.3, -0.2, 0.6, 0.0)),
            Pose3.exp((0.0, 0.0, 0.2, 0.1, -0.4, 0.3)),
            Pose3.exp((0.0, 0.0, 0.0, 0.3, -0.2, 0.1)),
        ]
        with self.assertRaises(ValueError):
            calibrate_extrinsic(make_observations(motions=motions))

    def test_two_axes_suffice(self) -> None:
        motions = [
            Pose3.exp((0.4, 0.0, 0.0, 0.5, 0.2, 0.1)),
            Pose3.exp((0.0, 0.3, 0.0, -0.2, 0.6, 0.0)),
            Pose3.exp((0.2, 0.1, 0.0, 0.1, -0.4, 0.3)),
        ]
        result = calibrate_extrinsic(make_observations(motions=motions))
        self.assertTrue(result.converged)


class TypeValidationTest(unittest.TestCase):
    def test_non_iterable_observations(self) -> None:
        with self.assertRaises(TypeError):
            calibrate_extrinsic(42)
        with self.assertRaises(TypeError):
            calibrate_extrinsic("not observations")

    def test_malformed_records(self) -> None:
        good = make_observations()
        with self.assertRaises(TypeError):
            calibrate_extrinsic([good[0], good[1], 7])
        with self.assertRaises(TypeError):
            calibrate_extrinsic([good[0], good[1], good[2][:2]])
        with self.assertRaises(TypeError):
            calibrate_extrinsic([good[0], good[1], good[2] + (1.0,)])
        with self.assertRaises(TypeError):
            calibrate_extrinsic([good[0], good[1], "abc"])

    def test_non_pose_motions(self) -> None:
        good = make_observations()
        with self.assertRaises(TypeError):
            calibrate_extrinsic([good[0], good[1], ((1, 0, 0), good[2][1], 1.0)])
        with self.assertRaises(TypeError):
            calibrate_extrinsic([good[0], good[1], (good[2][0], None, 1.0)])

    def test_non_pose_initial_pose(self) -> None:
        with self.assertRaises(TypeError):
            calibrate_extrinsic(make_observations(), initial_pose="identity")

    def test_wrong_type_weight(self) -> None:
        good = make_observations()
        with self.assertRaises(TypeError):
            calibrate_extrinsic([good[0], good[1], (good[2][0], good[2][1], "1")])
        with self.assertRaises(TypeError):
            calibrate_extrinsic([good[0], good[1], (good[2][0], good[2][1], True)])

    def test_wrong_type_control_parameters(self) -> None:
        observations = make_observations()
        with self.assertRaises(TypeError):
            calibrate_extrinsic(observations, max_iterations=10.0)
        with self.assertRaises(TypeError):
            calibrate_extrinsic(observations, max_iterations=True)
        with self.assertRaises(TypeError):
            calibrate_extrinsic(observations, tolerance="1e-6")
        with self.assertRaises(TypeError):
            calibrate_extrinsic(observations, tolerance=False)


class ValueValidationTest(unittest.TestCase):
    def test_too_few_observations(self) -> None:
        observations = make_observations()
        with self.assertRaises(ValueError):
            calibrate_extrinsic(observations[:2])
        with self.assertRaises(ValueError):
            calibrate_extrinsic([])

    def test_non_positive_or_non_finite_weight(self) -> None:
        good = make_observations()
        for bad_weight in (0.0, -1.0, math.inf, -math.inf, math.nan):
            with self.assertRaises(ValueError, msg=f"weight={bad_weight}"):
                calibrate_extrinsic(
                    [good[0], good[1], (good[2][0], good[2][1], bad_weight)]
                )

    def test_out_of_range_control_parameters(self) -> None:
        observations = make_observations()
        with self.assertRaises(ValueError):
            calibrate_extrinsic(observations, max_iterations=0)
        with self.assertRaises(ValueError):
            calibrate_extrinsic(observations, max_iterations=-3)
        for bad_tolerance in (0.0, -1e-6, math.inf, math.nan):
            with self.assertRaises(ValueError, msg=f"tolerance={bad_tolerance}"):
                calibrate_extrinsic(observations, tolerance=bad_tolerance)


class ImportPurityTest(unittest.TestCase):
    def test_import_does_not_start_http_service(self) -> None:
        code = (
            "import sys; "
            "import mappilot.calibration; "
            "assert 'mappilot.server' not in sys.modules, sys.modules.keys(); "
            "assert 'mappilot.service' not in sys.modules, sys.modules.keys()"
        )
        subprocess.run([sys.executable, "-c", code], check=True)


if __name__ == "__main__":
    unittest.main()
