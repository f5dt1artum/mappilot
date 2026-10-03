import math
import unittest

from mappilot.geometry import Pose3
from mappilot.registration import PointToPlaneICPResult, point_to_plane_icp


# Cube corners: well spread in 3D, so a varied normal field fully
# constrains the 6-DoF pose increment.
SOURCE = [
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 1.0, 0.0),
    (1.0, 0.0, 1.0),
    (0.0, 1.0, 1.0),
    (1.0, 1.0, 1.0),
]

NORMALS = [
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 1.0, 0.0),
    (1.0, 0.0, 1.0),
    (0.0, 1.0, 1.0),
    (1.0, 1.0, 1.0),
    (-1.0, 1.0, 0.0),
]


def rotate(pose, vectors):
    r = pose.rotation
    return [
        tuple(
            r[i][0] * v[0] + r[i][1] * v[1] + r[i][2] * v[2] for i in range(3)
        )
        for v in vectors
    ]


def pose_close(test_case, actual, expected, tol=1e-9):
    test_case.assertEqual(len(actual.translation), len(expected.translation))
    for a, e in zip(actual.translation, expected.translation):
        test_case.assertAlmostEqual(a, e, delta=tol)
    # q and -q describe the same rotation.
    qa = actual.quaternion
    qe = expected.quaternion
    test_case.assertTrue(
        all(abs(a - e) <= tol for a, e in zip(qa, qe))
        or all(abs(a + e) <= tol for a, e in zip(qa, qe))
    )


def recompute_projection_rmse(pose, source, target, normals, correspondences):
    moved = pose.transform_points(source)
    total = 0.0
    for source_index, target_index in correspondences:
        delta = [moved[source_index][k] - target[target_index][k] for k in range(3)]
        normal = normals[target_index]
        length = math.sqrt(sum(c * c for c in normal))
        residual = sum(delta[k] * normal[k] for k in range(3)) / length
        total += residual * residual
    return math.sqrt(total / len(correspondences))


class PerfectAlignmentTest(unittest.TestCase):
    def test_identical_clouds_converge_to_identity(self):
        result = point_to_plane_icp(SOURCE, SOURCE, NORMALS)
        self.assertIsInstance(result, PointToPlaneICPResult)
        self.assertTrue(result.converged)
        self.assertGreaterEqual(result.iterations, 1)
        self.assertAlmostEqual(result.rmse, 0.0, delta=1e-12)
        pose_close(self, result.pose, Pose3.identity())
        self.assertEqual(
            result.correspondences,
            ((0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)),
        )

    def test_recovers_known_rigid_transform_from_good_initial_guess(self):
        truth = Pose3((0.3, -0.2, 0.1), (0.36, 0.48, 0.6, 0.6))
        target = truth.transform_points(SOURCE)
        normals = rotate(truth, NORMALS)
        # A small tangent-space perturbation of the truth keeps every
        # nearest-neighbour relationship intact.
        guess = truth.compose(Pose3.exp((0.01, -0.02, 0.015, 0.02, -0.01, 0.01)))
        result = point_to_plane_icp(SOURCE, target, normals, initial_pose=guess)
        self.assertTrue(result.converged)
        self.assertAlmostEqual(result.rmse, 0.0, delta=1e-9)
        pose_close(self, result.pose, truth, tol=1e-8)

    def test_pure_translation_recovered(self):
        # A translation small relative to point spacing is inside ICP's
        # nearest-neighbour basin from the identity pose.
        truth = Pose3((0.2, -0.15, 0.1))
        target = truth.transform_points(SOURCE)
        result = point_to_plane_icp(SOURCE, target, NORMALS)
        self.assertTrue(result.converged)
        self.assertAlmostEqual(result.rmse, 0.0, delta=1e-9)
        pose_close(self, result.pose, truth, tol=1e-8)

    def test_accepts_single_pass_iterables(self):
        def gen(points):
            yield from points

        target = [(p[0] + 0.2, p[1], p[2]) for p in SOURCE]
        result = point_to_plane_icp(gen(SOURCE), gen(target), gen(NORMALS))
        self.assertTrue(result.converged)
        pose_close(self, result.pose, Pose3((0.2, 0.0, 0.0)), tol=1e-9)


class NormalHandlingTest(unittest.TestCase):
    def test_positive_scaling_of_normals_does_not_change_result(self):
        truth = Pose3((0.2, -0.15, 0.1), (0.9, 0.1, -0.2, 0.15))
        target = truth.transform_points(SOURCE)
        normals = rotate(truth, NORMALS)
        # Powers of two scale exactly, so normalized normals are identical.
        scaled = [
            (2.0 ** index * n[0], 2.0 ** index * n[1], 2.0 ** index * n[2])
            for index, n in enumerate(normals)
        ]
        baseline = point_to_plane_icp(SOURCE, target, normals)
        result = point_to_plane_icp(SOURCE, target, scaled)
        self.assertEqual(baseline.pose, result.pose)
        self.assertEqual(baseline.correspondences, result.correspondences)
        self.assertEqual(baseline.iterations, result.iterations)
        self.assertAlmostEqual(baseline.rmse, result.rmse, delta=1e-15)

    def test_unnormalized_normals_are_accepted(self):
        truth = Pose3((0.2, -0.15, 0.1))
        target = truth.transform_points(SOURCE)
        scaled = [(3.0 * n[0], 3.0 * n[1], 3.0 * n[2]) for n in NORMALS]
        result = point_to_plane_icp(SOURCE, target, scaled)
        self.assertTrue(result.converged)
        pose_close(self, result.pose, truth, tol=1e-8)

    def test_deterministic_across_calls(self):
        truth = Pose3((0.3, -0.2, 0.1), (0.36, 0.48, 0.6, 0.6))
        target = list(truth.transform_points(SOURCE))
        normals = rotate(truth, NORMALS)
        guess = truth.compose(Pose3.exp((0.01, 0.02, -0.01, 0.0, 0.0, 0.0)))
        first = point_to_plane_icp(SOURCE, target, normals, initial_pose=guess)
        second = point_to_plane_icp(
            tuple(tuple(p) for p in SOURCE),
            tuple(tuple(p) for p in target),
            tuple(tuple(n) for n in normals),
            initial_pose=guess,
        )
        self.assertEqual(first.pose, second.pose)
        self.assertEqual(first.correspondences, second.correspondences)
        self.assertEqual(first.iterations, second.iterations)
        self.assertEqual(first.converged, second.converged)
        self.assertAlmostEqual(first.rmse, second.rmse, delta=1e-15)


class IterationAndConvergenceTest(unittest.TestCase):
    def test_exhausting_iterations_returns_false_without_raising(self):
        truth = Pose3((0.2, -0.15, 0.1))
        target = truth.transform_points(SOURCE)
        result = point_to_plane_icp(SOURCE, target, NORMALS, max_iterations=1)
        self.assertFalse(result.converged)
        self.assertEqual(result.iterations, 1)
        # One update already solves those exact translated pairs.
        self.assertAlmostEqual(result.rmse, 0.0, delta=1e-9)
        pose_close(self, result.pose, truth, tol=1e-8)

    def test_iteration_cap_is_respected(self):
        truth = Pose3(
            (0.4, -0.3, 0.2),
            (math.cos(0.125), 0.0, 0.0, math.sin(0.125)),
        )
        target = truth.transform_points(SOURCE)
        normals = rotate(truth, NORMALS)
        result = point_to_plane_icp(
            SOURCE, target, normals, max_iterations=2, tolerance=1e-12
        )
        self.assertFalse(result.converged)
        self.assertEqual(result.iterations, 2)

    def test_tolerance_stops_when_rmse_plateaus(self):
        result = point_to_plane_icp(SOURCE, SOURCE, NORMALS, tolerance=1.0)
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
        (1.0, 1.0, 0.0),  # 5
        (1.0, 0.0, 1.0),  # 6
        (0.0, 1.0, 1.0),  # 7
    ]

    def test_ties_prefer_smallest_target_index_and_targets_reusable(self):
        source = [
            (0.0, 0.0, 0.0),  # tied between targets 0 and 1 -> 0
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0),  # repeats target 2
            (1.0, 1.0, 0.0),
            (1.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
        ]
        result = point_to_plane_icp(source, self.TARGET, NORMALS)
        self.assertTrue(result.converged)
        self.assertEqual(
            result.correspondences,
            ((0, 0), (1, 2), (2, 3), (3, 4), (4, 2), (5, 5), (6, 6), (7, 7)),
        )
        target_indices = [pair[1] for pair in result.correspondences]
        self.assertNotIn(1, target_indices)
        self.assertEqual(target_indices.count(2), 2)
        self.assertAlmostEqual(result.rmse, 0.0, delta=1e-12)
        # rmse must describe the correspondences under the final pose.
        manual = recompute_projection_rmse(
            result.pose, source, self.TARGET, NORMALS, result.correspondences
        )
        self.assertAlmostEqual(result.rmse, manual, delta=1e-12)

    def test_correspondences_sorted_and_consistent_with_rmse(self):
        source = [
            (0.2, -0.3, 0.5), (1.1, 0.4, -0.2), (-0.7, 0.9, 0.1),
            (0.3, 0.6, 1.0), (-1.0, -0.2, -0.4), (0.8, 0.1, 0.6),
            (-0.4, 0.3, -0.9), (0.5, -0.8, 0.2),
        ]
        truth = Pose3((0.5, -0.4, 0.2), (0.36, 0.48, 0.6, 0.6))
        target = list(truth.transform_points(source))
        normals = rotate(truth, NORMALS)
        guess = truth.compose(Pose3.exp((0.01, -0.01, 0.01, 0.02, 0.0, 0.0)))
        result = point_to_plane_icp(source, target, normals, initial_pose=guess)
        source_indices = [pair[0] for pair in result.correspondences]
        self.assertEqual(source_indices, sorted(source_indices))
        self.assertEqual(len(set(source_indices)), len(source_indices))
        manual = recompute_projection_rmse(
            result.pose, source, target, normals, result.correspondences
        )
        self.assertAlmostEqual(result.rmse, manual, delta=1e-12)
        self.assertAlmostEqual(result.rmse, 0.0, delta=1e-9)


class DistanceGateTest(unittest.TestCase):
    def test_gate_drops_distant_pairs(self):
        source = SOURCE[:6] + [(5.0, 5.0, 5.0)]
        target = SOURCE[:6] + [(6.0, 6.0, 6.0)]
        normals = NORMALS[:6] + [(1.0, 1.0, 1.0)]
        result = point_to_plane_icp(
            source, target, normals, max_correspondence_distance=0.5
        )
        self.assertTrue(result.converged)
        self.assertEqual(
            result.correspondences,
            ((0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)),
        )
        self.assertAlmostEqual(result.rmse, 0.0, delta=1e-12)

    def test_gate_too_strict_raises_value_error(self):
        target = [(p[0] + 10.0, p[1], p[2]) for p in SOURCE]
        with self.assertRaises(ValueError):
            point_to_plane_icp(
                SOURCE, target, NORMALS, max_correspondence_distance=0.1
            )


class DegeneracyTest(unittest.TestCase):
    def test_parallel_normals_on_plane_raise_value_error(self):
        # A flat patch with a parallel normal field leaves rotation about
        # the in-plane axes and in-plane translation unconstrained.
        source = [
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
            (1.0, 1.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0),
        ]
        target = [(p[0], p[1], p[2] + 0.5) for p in source]
        normals = [(0.0, 0.0, 1.0)] * 6
        with self.assertRaises(ValueError):
            point_to_plane_icp(source, target, normals)

    def test_parallel_normals_on_spread_points_raise_value_error(self):
        # Even with 3D-spread points, a parallel normal field leaves
        # translation along the two axes orthogonal to the normals (and
        # rotation about the normal axis) unconstrained.
        target = [(p[0], p[1], p[2] + 0.5) for p in SOURCE]
        normals = [(0.0, 0.0, 1.0)] * len(SOURCE)
        with self.assertRaises(ValueError):
            point_to_plane_icp(SOURCE, target, normals)


class ImmutabilityTest(unittest.TestCase):
    def test_result_is_immutable(self):
        result = point_to_plane_icp(SOURCE, SOURCE, NORMALS)
        with self.assertRaises(AttributeError):
            result.pose = Pose3()  # type: ignore[misc]
        with self.assertRaises(AttributeError):
            result.rmse = 1.0  # type: ignore[misc]
        self.assertIsInstance(result.correspondences, tuple)


class ValidationTest(unittest.TestCase):
    GOOD = SOURCE
    NORMALS_GOOD = NORMALS

    def test_non_iterable_cloud_raises_type_error(self):
        with self.assertRaises(TypeError):
            point_to_plane_icp(3, self.GOOD, self.NORMALS_GOOD)
        with self.assertRaises(TypeError):
            point_to_plane_icp("abc", self.GOOD, self.NORMALS_GOOD)
        with self.assertRaises(TypeError):
            point_to_plane_icp(self.GOOD, None, self.NORMALS_GOOD)

    def test_non_iterable_normals_raise_type_error(self):
        with self.assertRaises(TypeError):
            point_to_plane_icp(self.GOOD, self.GOOD, 3)
        with self.assertRaises(TypeError):
            point_to_plane_icp(self.GOOD, self.GOOD, "abcdefgh")

    def test_non_iterable_points_raise_type_error(self):
        with self.assertRaises(TypeError):
            point_to_plane_icp([1, 2, 3, 4, 5, 6], self.GOOD, self.NORMALS_GOOD)

    def test_wrong_point_dimension_raises_type_error(self):
        with self.assertRaises(TypeError):
            point_to_plane_icp(
                [(0.0, 0.0)] * 6, self.GOOD, self.NORMALS_GOOD
            )
        with self.assertRaises(TypeError):
            point_to_plane_icp(
                [(0.0, 0.0, 0.0, 0.0)] * 6, self.GOOD, self.NORMALS_GOOD
            )

    def test_wrong_normal_dimension_raises_type_error(self):
        with self.assertRaises(TypeError):
            point_to_plane_icp(self.GOOD, self.GOOD, [(0.0, 1.0)] * 8)
        with self.assertRaises(TypeError):
            point_to_plane_icp(self.GOOD, self.GOOD, [(0.0, 1.0, 0.0, 0.0)] * 8)

    def test_non_real_coordinates_raise_type_error(self):
        bad = [(0.0, "x", 0.0)] + self.GOOD[1:]
        with self.assertRaises(TypeError):
            point_to_plane_icp(bad, self.GOOD, self.NORMALS_GOOD)
        bad = [(None, 0.0, 0.0)] + self.GOOD[1:]
        with self.assertRaises(TypeError):
            point_to_plane_icp(bad, self.GOOD, self.NORMALS_GOOD)

    def test_non_real_normal_components_raise_type_error(self):
        bad = [(0.0, "x", 0.0)] + self.NORMALS_GOOD[1:]
        with self.assertRaises(TypeError):
            point_to_plane_icp(self.GOOD, self.GOOD, bad)

    def test_bool_coordinates_raise_type_error(self):
        bad = [(True, 0.0, 0.0)] + self.GOOD[1:]
        with self.assertRaises(TypeError):
            point_to_plane_icp(bad, self.GOOD, self.NORMALS_GOOD)
        bad_normals = [(False, 0.0, 1.0)] + self.NORMALS_GOOD[1:]
        with self.assertRaises(TypeError):
            point_to_plane_icp(self.GOOD, self.GOOD, bad_normals)

    def test_non_finite_coordinates_raise_value_error(self):
        for bad_value in (math.nan, math.inf, -math.inf):
            bad = [(bad_value, 0.0, 0.0)] + self.GOOD[1:]
            with self.assertRaises(ValueError):
                point_to_plane_icp(bad, self.GOOD, self.NORMALS_GOOD)
            bad_target = [(0.0, 0.0, bad_value)] + self.GOOD[1:]
            with self.assertRaises(ValueError):
                point_to_plane_icp(self.GOOD, bad_target, self.NORMALS_GOOD)
            bad_normals = [(0.0, bad_value, 0.0)] + self.NORMALS_GOOD[1:]
            with self.assertRaises(ValueError):
                point_to_plane_icp(self.GOOD, self.GOOD, bad_normals)

    def test_too_few_points_raise_value_error(self):
        with self.assertRaises(ValueError):
            point_to_plane_icp(self.GOOD[:5], self.GOOD, self.NORMALS_GOOD)
        with self.assertRaises(ValueError):
            point_to_plane_icp(self.GOOD, self.GOOD[:5], self.NORMALS_GOOD[:5])
        with self.assertRaises(ValueError):
            point_to_plane_icp([], self.GOOD, self.NORMALS_GOOD)

    def test_normal_count_mismatch_raises_value_error(self):
        with self.assertRaises(ValueError):
            point_to_plane_icp(self.GOOD, self.GOOD, self.NORMALS_GOOD[:7])
        with self.assertRaises(ValueError):
            point_to_plane_icp(
                self.GOOD, self.GOOD, self.NORMALS_GOOD + [(1.0, 0.0, 0.0)]
            )

    def test_zero_length_normal_raises_value_error(self):
        bad = [(0.0, 0.0, 0.0)] + self.NORMALS_GOOD[1:]
        with self.assertRaises(ValueError):
            point_to_plane_icp(self.GOOD, self.GOOD, bad)

    def test_initial_pose_type(self):
        with self.assertRaises(TypeError):
            point_to_plane_icp(
                self.GOOD, self.GOOD, self.NORMALS_GOOD,
                initial_pose=(0.0, 0.0, 0.0),
            )
        with self.assertRaises(TypeError):
            point_to_plane_icp(
                self.GOOD, self.GOOD, self.NORMALS_GOOD, initial_pose="pose"
            )
        # None and genuine Pose3 values are accepted.
        point_to_plane_icp(self.GOOD, self.GOOD, self.NORMALS_GOOD, initial_pose=None)
        point_to_plane_icp(
            self.GOOD, self.GOOD, self.NORMALS_GOOD, initial_pose=Pose3()
        )

    def test_max_iterations_validation(self):
        for bad in (1.0, True, "5", 2.5):
            with self.assertRaises(TypeError):
                point_to_plane_icp(
                    self.GOOD, self.GOOD, self.NORMALS_GOOD, max_iterations=bad
                )
        for bad in (0, -1):
            with self.assertRaises(ValueError):
                point_to_plane_icp(
                    self.GOOD, self.GOOD, self.NORMALS_GOOD, max_iterations=bad
                )

    def test_tolerance_validation(self):
        for bad in (0.0, -1e-6, math.inf, math.nan):
            with self.assertRaises(ValueError):
                point_to_plane_icp(
                    self.GOOD, self.GOOD, self.NORMALS_GOOD, tolerance=bad
                )
        for bad in (True, "1e-6", None):
            with self.assertRaises(TypeError):
                point_to_plane_icp(
                    self.GOOD, self.GOOD, self.NORMALS_GOOD, tolerance=bad
                )

    def test_gate_validation(self):
        for bad in (0.0, -1.0, math.inf, math.nan):
            with self.assertRaises(ValueError):
                point_to_plane_icp(
                    self.GOOD, self.GOOD, self.NORMALS_GOOD,
                    max_correspondence_distance=bad,
                )
        for bad in (True, "1.0"):
            with self.assertRaises(TypeError):
                point_to_plane_icp(
                    self.GOOD, self.GOOD, self.NORMALS_GOOD,
                    max_correspondence_distance=bad,
                )
        # Integral but non-boolean values are accepted, like other scalars.
        point_to_plane_icp(
            self.GOOD, self.GOOD, self.NORMALS_GOOD,
            max_correspondence_distance=10,
        )


if __name__ == "__main__":
    unittest.main()
