import math
import unittest

from mappilot.geometry import Pose3
from mappilot.registration import ICPResult, point_to_point_icp


def rotation_z(angle):
    return Pose3(
        (0.0, 0.0, 0.0),
        (math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0)),
    )


def pose_close(test_case, actual, expected, tol=1e-9):
    test_case.assertEqual(
        len(actual.translation), len(expected.translation)
    )
    for a, e in zip(actual.translation, expected.translation):
        test_case.assertAlmostEqual(a, e, delta=tol)
    # q and -q describe the same rotation.
    qa = actual.quaternion
    qe = expected.quaternion
    test_case.assertTrue(
        all(abs(a - e) <= tol for a, e in zip(qa, qe))
        or all(abs(a + e) <= tol for a, e in zip(qa, qe))
    )


def recompute_rmse(pose, source, target, correspondences, gate=None):
    moved = pose.transform_points(source)
    total = 0.0
    for source_index, target_index in correspondences:
        delta = [moved[source_index][k] - target[target_index][k] for k in range(3)]
        distance = math.sqrt(sum(d * d for d in delta))
        if gate is not None:
            assert distance <= gate + 1e-12
        total += distance * distance
    return math.sqrt(total / len(correspondences))


class PerfectAlignmentTest(unittest.TestCase):
    def test_identical_clouds_converge_to_identity(self):
        source = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
        result = point_to_point_icp(source, source)
        self.assertIsInstance(result, ICPResult)
        self.assertTrue(result.converged)
        self.assertGreaterEqual(result.iterations, 1)
        self.assertAlmostEqual(result.rmse, 0.0, delta=1e-12)
        pose_close(self, result.pose, Pose3.identity())
        self.assertEqual(
            result.correspondences, ((0, 0), (1, 1), (2, 2), (3, 3))
        )

    def test_recovers_known_rigid_transform_from_good_initial_guess(self):
        source = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.5, 0.5, 0.5),
        ]
        truth = Pose3((0.8, -0.6, 0.4), (0.36, 0.48, 0.6, 0.6))
        target = truth.transform_points(source)
        # A small tangent-space perturbation of the truth keeps every
        # nearest-neighbour relationship intact.
        guess = truth.compose(
            Pose3.exp((0.01, -0.02, 0.015, 0.02, -0.01, 0.01))
        )
        result = point_to_point_icp(source, target, initial_pose=guess)
        self.assertTrue(result.converged)
        self.assertAlmostEqual(result.rmse, 0.0, delta=1e-9)
        pose_close(self, result.pose, truth, tol=1e-8)

    def test_pure_translation_recovered(self):
        source = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                  (0.0, 0.0, 1.0)]
        # A translation small relative to point spacing is inside ICP's
        # nearest-neighbour basin from the identity pose.
        truth = Pose3((0.2, -0.15, 0.1))
        target = truth.transform_points(source)
        result = point_to_point_icp(source, target)
        self.assertTrue(result.converged)
        self.assertAlmostEqual(result.rmse, 0.0, delta=1e-9)
        pose_close(self, result.pose, truth, tol=1e-8)

    def test_large_translation_recovered_from_matching_initial_pose(self):
        source = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                  (0.0, 0.0, 1.0)]
        truth = Pose3((2.0, -3.0, 1.0))
        target = truth.transform_points(source)
        result = point_to_point_icp(
            source, target, initial_pose=Pose3((1.9, -2.9, 1.05))
        )
        self.assertTrue(result.converged)
        self.assertAlmostEqual(result.rmse, 0.0, delta=1e-9)
        pose_close(self, result.pose, truth, tol=1e-8)

    def test_deterministic_across_calls(self):
        source = [
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0), (-0.5, -0.5, 0.2),
        ]
        truth = Pose3((0.3, -0.2, 0.1), (0.36, 0.48, 0.6, 0.6))
        target = truth.transform_points(source)
        guess = truth.compose(Pose3.exp((0.01, 0.02, -0.01, 0.0, 0.0, 0.0)))
        first = point_to_point_icp(source, target, initial_pose=guess)
        second = point_to_point_icp(
            tuple(tuple(p) for p in source), target, initial_pose=guess
        )
        self.assertEqual(first.pose, second.pose)
        self.assertEqual(first.correspondences, second.correspondences)
        self.assertEqual(first.iterations, second.iterations)
        self.assertEqual(first.converged, second.converged)
        self.assertAlmostEqual(first.rmse, second.rmse, delta=1e-15)

    def test_accepts_single_pass_iterables(self):
        def gen(points):
            yield from points

        source = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                  (0.0, 0.0, 1.0)]
        target = [(p[0] + 0.2, p[1], p[2]) for p in source]
        result = point_to_point_icp(gen(source), gen(target))
        self.assertTrue(result.converged)
        pose_close(self, result.pose, Pose3((0.2, 0.0, 0.0)), tol=1e-9)


class IterationAndConvergenceTest(unittest.TestCase):
    def test_exhausting_iterations_returns_false_without_raising(self):
        source = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                  (0.0, 0.0, 1.0)]
        target = [(0.3, 0.0, 0.0), (1.3, 0.0, 0.0), (0.3, 1.0, 0.0),
                  (0.3, 0.0, 1.0)]
        result = point_to_point_icp(source, target, max_iterations=1)
        self.assertFalse(result.converged)
        self.assertEqual(result.iterations, 1)
        # One update already solves those exact translated pairs.
        self.assertAlmostEqual(result.rmse, 0.0, delta=1e-12)
        pose_close(self, result.pose, Pose3((0.3, 0.0, 0.0)), tol=1e-9)

    def test_iteration_cap_is_respected(self):
        source = [
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0), (0.5, 0.5, 0.5),
        ]
        truth = Pose3((0.4, -0.3, 0.2), rotation_z(0.25).quaternion)
        target = truth.transform_points(source)
        result = point_to_point_icp(source, target, max_iterations=2)
        self.assertFalse(result.converged)
        self.assertEqual(result.iterations, 2)

    def test_tolerance_stops_when_rmse_plateaus(self):
        source = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        target = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        result = point_to_point_icp(source, target, tolerance=1.0)
        self.assertTrue(result.converged)


class CorrespondenceTest(unittest.TestCase):
    # t0 and t1 are coincident, so a source sitting on them is tied forever
    # regardless of the pose; all sources coincide with targets at identity
    # (zero residual), so ICP converges there with deterministic ties.
    TARGET = [
        (0.0, 0.0, 0.0),  # 0: coincident with 1
        (0.0, 0.0, 0.0),  # 1: tie loser, must never be selected
        (1.0, 0.0, 0.0),  # 2: reused by two sources
        (0.0, 1.0, 0.0),  # 3
        (0.0, 0.0, 1.0),  # 4
    ]

    def test_ties_prefer_smallest_target_index_and_targets_reusable(self):
        source = [
            (0.0, 0.0, 0.0),  # tied between targets 0 and 1 -> 0
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0),  # repeats target 2
        ]
        result = point_to_point_icp(source, self.TARGET)
        self.assertTrue(result.converged)
        self.assertEqual(
            result.correspondences, ((0, 0), (1, 2), (2, 3), (3, 4), (4, 2))
        )
        target_indices = [pair[1] for pair in result.correspondences]
        self.assertNotIn(1, target_indices)
        self.assertEqual(target_indices.count(2), 2)
        self.assertAlmostEqual(result.rmse, 0.0, delta=1e-12)
        # rmse must describe the correspondences under the final pose.
        manual = recompute_rmse(
            result.pose, source, self.TARGET, result.correspondences
        )
        self.assertAlmostEqual(result.rmse, manual, delta=1e-12)

    def test_correspondences_sorted_and_consistent_with_rmse(self):
        source = [
            (0.2, -0.3, 0.5), (1.1, 0.4, -0.2), (-0.7, 0.9, 0.1),
            (0.3, 0.6, 1.0), (-1.0, -0.2, -0.4),
        ]
        truth = Pose3((0.5, -0.4, 0.2), (0.36, 0.48, 0.6, 0.6))
        target = list(truth.transform_points(source))
        target.append((5.0, 5.0, 5.0))  # extra unrepeated decoy target
        guess = truth.compose(Pose3.exp((0.01, -0.01, 0.01, 0.02, 0.0, 0.0)))
        result = point_to_point_icp(source, target, initial_pose=guess)
        source_indices = [pair[0] for pair in result.correspondences]
        self.assertEqual(source_indices, sorted(source_indices))
        self.assertEqual(len(set(source_indices)), len(source_indices))
        manual = recompute_rmse(
            result.pose, source, target, result.correspondences
        )
        self.assertAlmostEqual(result.rmse, manual, delta=1e-12)
        self.assertAlmostEqual(result.rmse, 0.0, delta=1e-9)

    def test_result_rotation_has_unit_determinant(self):
        source = [
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ]
        target = [
            (2.0, 0.0, 0.0), (2.0, 1.0, 0.0), (2.0, 0.0, 1.0),
            (3.0, -1.0, -1.0),
        ]
        result = point_to_point_icp(source, target)
        r = result.pose.rotation
        determinant = (
            r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
            - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
            + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0])
        )
        self.assertAlmostEqual(determinant, 1.0, delta=1e-9)


class DistanceGateTest(unittest.TestCase):
    def test_gate_drops_distant_pairs(self):
        source = [
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
            (5.0, 5.0, 5.0),
        ]
        target = [
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
            (6.0, 6.0, 6.0),
        ]
        result = point_to_point_icp(
            source, target, max_correspondence_distance=0.5
        )
        self.assertTrue(result.converged)
        self.assertEqual(result.correspondences, ((0, 0), (1, 1), (2, 2)))
        self.assertAlmostEqual(result.rmse, 0.0, delta=1e-12)

    def test_gate_too_strict_raises_value_error(self):
        source = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        target = [(10.0, 0.0, 0.0), (11.0, 0.0, 0.0), (10.0, 1.0, 0.0)]
        with self.assertRaises(ValueError):
            point_to_point_icp(source, target, max_correspondence_distance=0.1)


class DegeneracyTest(unittest.TestCase):
    def test_collinear_correspondences_raise_value_error(self):
        source = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
        target = [(10.0, 0.0, 0.0), (11.0, 0.0, 0.0), (12.0, 0.0, 0.0)]
        with self.assertRaises(ValueError):
            point_to_point_icp(source, target)

    def test_collinear_correspondences_with_translation_raise_value_error(self):
        source = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
        target = [(10.0, 0.0, 0.0), (11.0, 0.0, 0.0), (12.0, 0.0, 0.0)]
        # Aligned along the line, the three pairs are distinct but collinear,
        # so the rotation is not uniquely determined.
        with self.assertRaises(ValueError):
            point_to_point_icp(source, target, initial_pose=Pose3((10.0, 0.0, 0.0)))


class ImmutabilityTest(unittest.TestCase):
    def test_result_is_immutable(self):
        result = point_to_point_icp(
            [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        )
        with self.assertRaises(AttributeError):
            result.pose = Pose3()  # type: ignore[misc]
        with self.assertRaises(AttributeError):
            result.rmse = 1.0  # type: ignore[misc]
        self.assertIsInstance(result.correspondences, tuple)


class ValidationTest(unittest.TestCase):
    GOOD = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]

    def test_non_iterable_cloud_raises_type_error(self):
        with self.assertRaises(TypeError):
            point_to_point_icp(3, self.GOOD)
        with self.assertRaises(TypeError):
            point_to_point_icp("abc", self.GOOD)

    def test_non_iterable_points_raise_type_error(self):
        with self.assertRaises(TypeError):
            point_to_point_icp([1, 2, 3], self.GOOD)

    def test_wrong_point_dimension_raises_type_error(self):
        with self.assertRaises(TypeError):
            point_to_point_icp([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)], self.GOOD)
        with self.assertRaises(TypeError):
            point_to_point_icp(
                [(0.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
                self.GOOD,
            )

    def test_non_real_coordinates_raise_type_error(self):
        with self.assertRaises(TypeError):
            point_to_point_icp([(0.0, "x", 0.0), (1.0, 0.0, 0.0),
                                (0.0, 1.0, 0.0)], self.GOOD)
        with self.assertRaises(TypeError):
            point_to_point_icp([(None, 0.0, 0.0), (1.0, 0.0, 0.0),
                                (0.0, 1.0, 0.0)], self.GOOD)

    def test_bool_coordinates_raise_type_error(self):
        with self.assertRaises(TypeError):
            point_to_point_icp([(True, 0.0, 0.0), (1.0, 0.0, 0.0),
                                (0.0, 1.0, 0.0)], self.GOOD)

    def test_non_finite_coordinates_raise_value_error(self):
        for bad in (math.nan, math.inf, -math.inf):
            with self.assertRaises(ValueError):
                point_to_point_icp(
                    [(bad, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
                    self.GOOD,
                )
            with self.assertRaises(ValueError):
                point_to_point_icp(
                    self.GOOD,
                    [(0.0, 0.0, bad), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
                )

    def test_too_few_points_raise_value_error(self):
        with self.assertRaises(ValueError):
            point_to_point_icp([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)], self.GOOD)
        with self.assertRaises(ValueError):
            point_to_point_icp([], self.GOOD)

    def test_initial_pose_type(self):
        with self.assertRaises(TypeError):
            point_to_point_icp(self.GOOD, self.GOOD, initial_pose=(0.0, 0.0, 0.0))
        with self.assertRaises(TypeError):
            point_to_point_icp(self.GOOD, self.GOOD, initial_pose="pose")
        # None and genuine Pose3 values are accepted.
        point_to_point_icp(self.GOOD, self.GOOD, initial_pose=None)
        point_to_point_icp(self.GOOD, self.GOOD, initial_pose=Pose3())

    def test_max_iterations_validation(self):
        for bad in (1.0, True, "5", 2.5):
            with self.assertRaises(TypeError):
                point_to_point_icp(self.GOOD, self.GOOD, max_iterations=bad)
        for bad in (0, -1):
            with self.assertRaises(ValueError):
                point_to_point_icp(self.GOOD, self.GOOD, max_iterations=bad)

    def test_tolerance_validation(self):
        for bad in (0.0, -1e-6, math.inf, math.nan):
            with self.assertRaises(ValueError):
                point_to_point_icp(self.GOOD, self.GOOD, tolerance=bad)
        for bad in (True, "1e-6", None):
            with self.assertRaises(TypeError):
                point_to_point_icp(self.GOOD, self.GOOD, tolerance=bad)

    def test_gate_validation(self):
        for bad in (0.0, -1.0, math.inf, math.nan):
            with self.assertRaises(ValueError):
                point_to_point_icp(
                    self.GOOD, self.GOOD, max_correspondence_distance=bad
                )
        for bad in (True, "1.0"):
            with self.assertRaises(TypeError):
                point_to_point_icp(
                    self.GOOD, self.GOOD, max_correspondence_distance=bad
                )
        # Integral but non-boolean values are accepted, like other scalars.
        point_to_point_icp(
            self.GOOD, self.GOOD, max_correspondence_distance=10
        )


if __name__ == "__main__":
    unittest.main()
