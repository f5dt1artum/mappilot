import inspect
import math
import sys
import unittest

from mappilot.geometry import Pose3
from mappilot.registration import ICPResult, point_to_point_icp


def quaternion_close(test_case, actual, expected, tol=1e-9):
    # q and -q represent the same rotation; compare against both signs.
    direct = max(abs(a - e) for a, e in zip(actual, expected))
    flipped = max(abs(a + e) for a, e in zip(actual, expected))
    test_case.assertLessEqual(min(direct, flipped), tol)


# Generic, asymmetric (non-degenerate, unique nearest neighbours) clouds.
SOURCE = [
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 2.0, 0.0),
    (0.0, 0.0, 3.0),
    (1.5, 0.7, -0.4),
    (-0.6, 1.1, 2.3),
]
GROUND_TRUTH = Pose3((0.4, -0.8, 1.2), (0.36, 0.48, 0.6, 0.6))
TARGET = [GROUND_TRUTH.transform_point(point) for point in SOURCE]


def manual_rmse(pose, source, target, correspondences):
    transformed = pose.transform_points(source)
    total = 0.0
    for source_index, target_index in correspondences:
        sp = transformed[source_index]
        tp = target[target_index]
        total += sum((sp[axis] - tp[axis]) ** 2 for axis in range(3))
    return math.sqrt(total / len(correspondences))


class ExactRecoveryTest(unittest.TestCase):
    def test_recovers_transform_from_exact_initial_pose(self):
        result = point_to_point_icp(SOURCE, TARGET, initial_pose=GROUND_TRUTH)
        self.assertIsInstance(result, ICPResult)
        self.assertTrue(result.converged)
        self.assertEqual(result.rmse, 0.0)
        self.assertEqual(result.pose, GROUND_TRUTH)
        self.assertEqual(
            result.correspondences, tuple((index, index) for index in range(len(SOURCE)))
        )

    def test_recovers_transform_from_nearby_initial_pose(self):
        perturb = Pose3.exp((0.02, -0.01, 0.015, 0.03, -0.02, 0.01))
        initial = perturb.compose(GROUND_TRUTH)
        result = point_to_point_icp(SOURCE, TARGET, initial_pose=initial)
        self.assertTrue(result.converged)
        self.assertLess(result.rmse, 1e-10)
        for axis in range(3):
            self.assertAlmostEqual(
                result.pose.translation[axis], GROUND_TRUTH.translation[axis], delta=1e-9
            )
        quaternion_close(self, result.pose.quaternion, GROUND_TRUTH.quaternion)

    def test_recovers_with_minimum_three_points(self):
        triangle = [(-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (0.0, 1.5, 0.0)]
        pose = Pose3((1.0, -1.0, 0.5), (0.36, 0.48, 0.6, 0.6))
        target = [pose.transform_point(point) for point in triangle]
        result = point_to_point_icp(triangle, target, initial_pose=pose)
        self.assertTrue(result.converged)
        self.assertLess(result.rmse, 1e-10)
        self.assertEqual(result.pose, pose)

    def test_self_alignment_is_identity(self):
        result = point_to_point_icp(SOURCE, SOURCE)
        self.assertTrue(result.converged)
        self.assertEqual(result.iterations, 1)
        self.assertEqual(result.rmse, 0.0)
        self.assertEqual(result.pose, Pose3.identity())
        self.assertEqual(
            result.correspondences, tuple((i, i) for i in range(len(SOURCE)))
        )


class IterationSemanticsTest(unittest.TestCase):
    def test_iterations_count_actual_updates_and_run_is_deterministic(self):
        result = point_to_point_icp(SOURCE, TARGET, initial_pose=GROUND_TRUTH)
        self.assertGreaterEqual(result.iterations, 1)
        again = point_to_point_icp(SOURCE, TARGET, initial_pose=GROUND_TRUTH)
        self.assertEqual(again.pose, result.pose)
        self.assertEqual(again.converged, result.converged)
        self.assertEqual(again.iterations, result.iterations)
        self.assertEqual(again.rmse, result.rmse)
        self.assertEqual(again.correspondences, result.correspondences)

    def test_exhausting_budget_returns_last_result_without_raising(self):
        result = point_to_point_icp(SOURCE, TARGET, max_iterations=1)
        self.assertFalse(result.converged)
        self.assertEqual(result.iterations, 1)
        self.assertIsInstance(result.pose, Pose3)
        self.assertGreaterEqual(result.rmse, 0.0)

    def test_iterations_never_exceed_budget(self):
        helix = [
            (float(i), math.sin(i) * 0.5, math.cos(i) * 0.7) for i in range(8)
        ]
        pose = Pose3.exp((0.2, 0.1, -0.15, 1.0, -0.5, 0.3))
        target = [pose.transform_point(point) for point in helix]
        result = point_to_point_icp(helix, target, max_iterations=2)
        self.assertEqual(result.iterations, 2)
        self.assertFalse(result.converged)

    def test_loose_tolerance_converges_after_one_update(self):
        result = point_to_point_icp(SOURCE, TARGET, tolerance=100.0)
        self.assertTrue(result.converged)
        self.assertEqual(result.iterations, 1)

    def test_default_parameters_match_contract(self):
        parameters = inspect.signature(point_to_point_icp).parameters
        self.assertIsNone(parameters["initial_pose"].default)
        self.assertEqual(parameters["max_iterations"].default, 50)
        self.assertEqual(parameters["tolerance"].default, 1e-6)
        self.assertIsNone(parameters["max_correspondence_distance"].default)


class ResultConsistencyTest(unittest.TestCase):
    def test_rmse_matches_final_pose_and_correspondences(self):
        result = point_to_point_icp(SOURCE, TARGET, max_iterations=1)
        expected = manual_rmse(result.pose, SOURCE, TARGET, result.correspondences)
        self.assertAlmostEqual(result.rmse, expected, delta=1e-12)

    def test_correspondences_sorted_by_source_index(self):
        result = point_to_point_icp(SOURCE, TARGET, max_iterations=3)
        source_indices = [pair[0] for pair in result.correspondences]
        self.assertEqual(source_indices, sorted(source_indices))
        self.assertEqual(len(source_indices), len(set(source_indices)))
        for source_index, target_index in result.correspondences:
            self.assertIsInstance(source_index, int)
            self.assertIsInstance(target_index, int)

    def test_result_is_immutable(self):
        result = point_to_point_icp(SOURCE, TARGET, initial_pose=GROUND_TRUTH)
        with self.assertRaises(AttributeError):
            result.pose = GROUND_TRUTH  # type: ignore[misc]
        with self.assertRaises(AttributeError):
            result.converged = False  # type: ignore[misc]
        with self.assertRaises(AttributeError):
            result.correspondences.append((0, 0))  # type: ignore[attr-defined]
        # Positional NamedTuple layout: pose, converged, iterations, rmse, corrs.
        self.assertIs(result[0], result.pose)
        self.assertIs(result[1], result.converged)
        self.assertEqual(result[2], result.iterations)
        self.assertEqual(result[3], result.rmse)
        self.assertIs(result[4], result.correspondences)


class AssociationTest(unittest.TestCase):
    def test_target_points_may_be_chosen_repeatedly(self):
        source = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
        target = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        result = point_to_point_icp(source, target, max_iterations=3)
        chosen_targets = [target_index for _, target_index in result.correspondences]
        self.assertEqual(result.correspondences, ((0, 0), (1, 1), (2, 2), (3, 0)))
        self.assertEqual(chosen_targets.count(0), 2)

    def test_ties_resolve_to_smallest_target_index(self):
        source = [
            (0.0, 0.0, 0.0),
            (10.0, 0.0, 0.0),
            (0.0, 10.0, 0.0),
            (0.0, 0.0, 10.0),
        ]
        # Targets 0 and 1 coincide: the first source lands exactly on both,
        # so target 0 must win the zero-distance tie.
        target = [
            (1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (11.0, 0.0, 0.0),
            (1.0, 10.0, 0.0),
            (1.0, 0.0, 10.0),
        ]
        result = point_to_point_icp(
            source, target, initial_pose=Pose3((1.0, 0.0, 0.0)), max_iterations=2
        )
        self.assertEqual(
            result.correspondences, ((0, 0), (1, 2), (2, 3), (3, 4))
        )

    def test_gate_keeps_pairs_at_or_below_distance(self):
        source = [(0.0, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, 0.0, 10.0)]
        target = [(2.0, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, 0.0, 10.0)]
        at_boundary = point_to_point_icp(
            source, target, max_correspondence_distance=2.0, max_iterations=1
        )
        self.assertEqual(len(at_boundary.correspondences), 3)
        with self.assertRaises(ValueError):
            point_to_point_icp(
                source, target, max_correspondence_distance=1.999999, max_iterations=1
            )

    def test_wide_gate_matches_ungated_run(self):
        gated = point_to_point_icp(
            SOURCE, TARGET, initial_pose=GROUND_TRUTH, max_correspondence_distance=1000.0
        )
        ungated = point_to_point_icp(SOURCE, TARGET, initial_pose=GROUND_TRUTH)
        self.assertEqual(gated.pose, ungated.pose)
        self.assertEqual(gated.correspondences, ungated.correspondences)
        self.assertEqual(gated.rmse, ungated.rmse)

    def test_tight_gate_with_too_few_pairs_raises_value_error(self):
        source = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        target = [(10.0, 0.0, 0.0), (11.0, 0.0, 0.0), (10.0, 1.0, 0.0)]
        with self.assertRaises(ValueError):
            point_to_point_icp(source, target, max_correspondence_distance=1.0)

    def test_collinear_surviving_pairs_raise_value_error(self):
        # Three close collinear pairs survive the gate; the fourth source is
        # far enough away that the gate removes it, leaving a rank-1 fit.
        source = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (100.0, 100.0, 100.0),
        ]
        target = [
            (0.0, 0.1, 0.0),
            (1.0, 0.1, 0.0),
            (2.0, 0.1, 0.0),
        ]
        with self.assertRaises(ValueError):
            point_to_point_icp(source, target, max_correspondence_distance=1.0)


class CloudValidationTest(unittest.TestCase):
    def test_non_iterable_clouds_raise_type_error(self):
        with self.assertRaises(TypeError):
            point_to_point_icp(None, TARGET)
        with self.assertRaises(TypeError):
            point_to_point_icp(7, TARGET)
        with self.assertRaises(TypeError):
            point_to_point_icp(SOURCE, None)

    def test_points_must_be_three_dimensional(self):
        with self.assertRaises(TypeError):
            point_to_point_icp([(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)], TARGET)
        with self.assertRaises(TypeError):
            point_to_point_icp(
                [(1.0, 2.0, 3.0, 4.0)] * 3, TARGET
            )
        with self.assertRaises(TypeError):
            point_to_point_icp([1.0, 2.0, 3.0], TARGET)

    def test_coordinates_must_be_real_non_boolean(self):
        for bad in ("x", None, 1 + 2j, True, False):
            with self.subTest(bad=bad):
                cloud = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, bad)]
                with self.assertRaises(TypeError):
                    point_to_point_icp(cloud, TARGET)

    def test_non_finite_coordinates_raise_value_error(self):
        for bad in (math.nan, math.inf, -math.inf):
            with self.subTest(bad=bad):
                cloud = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, bad, 0.0)]
                with self.assertRaises(ValueError):
                    point_to_point_icp(cloud, TARGET)
                with self.assertRaises(ValueError):
                    point_to_point_icp(SOURCE, cloud)

    def test_clouds_need_at_least_three_points(self):
        for count in (0, 1, 2):
            cloud = SOURCE[:count]
            with self.subTest(count=count):
                with self.assertRaises(ValueError):
                    point_to_point_icp(cloud, TARGET)
                with self.assertRaises(ValueError):
                    point_to_point_icp(SOURCE, cloud)


class ParameterValidationTest(unittest.TestCase):
    def test_initial_pose_must_be_pose3(self):
        for bad in ((0.0, 0.0, 0.0), "identity", 42, object()):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    point_to_point_icp(SOURCE, TARGET, initial_pose=bad)

    def test_max_iterations_must_be_positive_integer(self):
        for bad in (1.0, True, False, "5", None, 1 + 0j):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    point_to_point_icp(SOURCE, TARGET, max_iterations=bad)
        for bad in (0, -1, -50):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    point_to_point_icp(SOURCE, TARGET, max_iterations=bad)

    def test_tolerance_must_be_finite_positive(self):
        for bad in ("1e-6", True, False, None):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    point_to_point_icp(SOURCE, TARGET, tolerance=bad)
        for bad in (0.0, -1.0, math.nan, math.inf, -math.inf):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    point_to_point_icp(SOURCE, TARGET, tolerance=bad)

    def test_gate_must_be_finite_positive_when_given(self):
        for bad in ("1.0", True, False):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    point_to_point_icp(SOURCE, TARGET, max_correspondence_distance=bad)
        for bad in (0.0, -1.0, math.nan, math.inf, -math.inf):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    point_to_point_icp(SOURCE, TARGET, max_correspondence_distance=bad)


class ImportSideEffectTest(unittest.TestCase):
    def test_importing_registration_does_not_start_service(self):
        self.assertNotIn("mappilot.server", sys.modules)


if __name__ == "__main__":
    unittest.main()
