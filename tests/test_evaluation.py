import math
import subprocess
import sys
import unittest

from mappilot.evaluation import TrajectoryEvaluationResult, evaluate_trajectory
from mappilot.geometry import Pose3


def rotation_z(angle):
    return Pose3(
        (0.0, 0.0, 0.0),
        (math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0)),
    )


def make_path(count=6):
    """A non-degenerate 3D reference trajectory with distinct rotations."""
    return [
        (
            10 * index,
            Pose3(
                (index * 1.0, (index % 2) * 0.5, index * index * 0.3),
                rotation_z(0.1 * index).quaternion,
            ),
        )
        for index in range(count)
    ]


def pose_close(test_case, actual, expected, tol=1e-9):
    for a, e in zip(actual.translation, expected.translation):
        test_case.assertAlmostEqual(a, e, delta=tol)
    qa = actual.quaternion
    qe = expected.quaternion
    test_case.assertTrue(
        all(abs(a - e) <= tol for a, e in zip(qa, qe))
        or all(abs(a + e) <= tol for a, e in zip(qa, qe))
    )


def rmse(errors):
    return math.sqrt(sum(e * e for e in errors) / len(errors))


class NoneAlignmentTest(unittest.TestCase):
    def test_identity_alignment_and_scale(self):
        reference = make_path()
        estimated = [(i, pose) for i, pose in reference]
        result = evaluate_trajectory(reference, estimated, alignment="none")
        self.assertIsInstance(result, TrajectoryEvaluationResult)
        pose_close(self, result.alignment_pose, Pose3.identity())
        self.assertEqual(result.scale, 1.0)
        self.assertEqual(result.matched_ids, (0, 10, 20, 30, 40, 50))
        self.assertEqual(result.aligned_poses, tuple(pose for _, pose in estimated))
        self.assertTrue(all(e == 0.0 for e in result.translation_errors))
        self.assertTrue(all(e == 0.0 for e in result.rotation_errors))

    def test_constant_offset_translation_error(self):
        reference = [(i, Pose3((float(i), 0.0, 0.0))) for i in range(4)]
        estimated = [(i, Pose3((float(i), 1.0, 0.0))) for i in range(4)]
        result = evaluate_trajectory(reference, estimated, alignment="none")
        self.assertEqual(result.translation_errors, (1.0, 1.0, 1.0, 1.0))
        self.assertAlmostEqual(result.translation_rmse, 1.0)
        self.assertTrue(all(e == 0.0 for e in result.rotation_errors))
        # Identical relative motion on both sides: zero relative error.
        self.assertEqual(len(result.relative_translation_errors), 3)
        self.assertTrue(all(e < 1e-12 for e in result.relative_translation_errors))
        self.assertTrue(all(e < 1e-12 for e in result.relative_rotation_errors))

    def test_degenerate_positions_do_not_fail(self):
        reference = [(i, Pose3((0.0, 0.0, 0.0))) for i in range(3)]
        estimated = [(i, Pose3((1.0, 1.0, 1.0))) for i in range(3)]
        result = evaluate_trajectory(reference, estimated, alignment="none")
        self.assertEqual(len(result.matched_ids), 3)
        self.assertTrue(all(abs(e - math.sqrt(3.0)) < 1e-12
                            for e in result.translation_errors))


class SE3AlignmentTest(unittest.TestCase):
    def test_recovers_known_rigid_transform(self):
        reference = make_path()
        truth = Pose3((0.8, -0.6, 0.4), (0.36, 0.48, 0.6, 0.6))
        inverse = truth.inverse()
        estimated = [
            (record_id, inverse.compose(pose)) for record_id, pose in reference
        ]
        result = evaluate_trajectory(reference, estimated, alignment="se3")
        self.assertEqual(result.scale, 1.0)
        pose_close(self, result.alignment_pose, truth, tol=1e-8)
        self.assertTrue(all(e < 1e-8 for e in result.translation_errors))
        self.assertTrue(all(e < 1e-8 for e in result.rotation_errors))
        self.assertLess(result.translation_rmse, 1e-8)
        self.assertLess(result.rotation_rmse, 1e-8)
        self.assertTrue(all(e < 1e-8 for e in result.relative_translation_errors))
        self.assertTrue(all(e < 1e-8 for e in result.relative_rotation_errors))

    def test_default_alignment_is_se3(self):
        reference = make_path()
        truth = Pose3((1.0, 2.0, 3.0), rotation_z(0.3).quaternion)
        inverse = truth.inverse()
        estimated = [
            (record_id, inverse.compose(pose)) for record_id, pose in reference
        ]
        result = evaluate_trajectory(reference, estimated)
        pose_close(self, result.alignment_pose, truth, tol=1e-8)
        self.assertEqual(result.scale, 1.0)

    def test_collinear_positions_raise(self):
        reference = make_path()
        estimated = [(record_id, Pose3((float(k), 0.0, 0.0)))
                     for k, (record_id, _) in enumerate(reference)]
        with self.assertRaises(ValueError):
            evaluate_trajectory(reference, estimated, alignment="se3")
        # Degenerate reference positions fail too.
        collapsed = [(record_id, Pose3((float(k), 0.0, 0.0)))
                     for k, (record_id, _) in enumerate(reference)]
        with self.assertRaises(ValueError):
            evaluate_trajectory(collapsed, make_path(), alignment="se3")


class Sim3AlignmentTest(unittest.TestCase):
    def test_recovers_known_scale_and_transform(self):
        reference = make_path()
        truth = Pose3((0.8, -0.6, 0.4), (0.36, 0.48, 0.6, 0.6))
        scale = 2.5
        inverse = truth.inverse()
        estimated = []
        for record_id, pose in reference:
            unscaled = inverse.compose(pose)
            estimated.append(
                (
                    record_id,
                    Pose3(
                        tuple(c / scale for c in unscaled.translation),
                        unscaled.quaternion,
                    ),
                )
            )
        result = evaluate_trajectory(reference, estimated, alignment="sim3")
        self.assertAlmostEqual(result.scale, scale, delta=1e-9)
        pose_close(self, result.alignment_pose, truth, tol=1e-8)
        self.assertTrue(all(e < 1e-8 for e in result.translation_errors))
        self.assertTrue(all(e < 1e-8 for e in result.rotation_errors))
        self.assertLess(result.relative_translation_rmse, 1e-8)
        self.assertLess(result.relative_rotation_rmse, 1e-8)

    def test_fitted_scale_is_strictly_positive(self):
        # Even when the reference is the point reflection of the estimated
        # positions (best explained by a mirror), the fitted scale stays
        # strictly positive.
        estimated = make_path()
        reference = [
            (record_id, Pose3(tuple(-c for c in pose.translation), pose.quaternion))
            for record_id, pose in estimated
        ]
        result = evaluate_trajectory(reference, estimated, alignment="sim3")
        self.assertGreater(result.scale, 0.0)
        self.assertTrue(math.isfinite(result.scale))

    def test_collinear_positions_raise(self):
        reference = make_path()
        estimated = [(record_id, Pose3((float(k), 0.0, 0.0)))
                     for k, (record_id, _) in enumerate(reference)]
        with self.assertRaises(ValueError):
            evaluate_trajectory(reference, estimated, alignment="sim3")


class MatchingTest(unittest.TestCase):
    def test_intersection_in_reference_order(self):
        reference = [(5, Pose3((0.0, 0.0, 0.0))),
                     (2, Pose3((1.0, 0.0, 0.0))),
                     (9, Pose3((0.0, 1.0, 0.0))),
                     (7, Pose3((0.0, 0.0, 1.0)))]
        estimated = [(7, Pose3((0.0, 0.0, 1.0))),
                     (2, Pose3((1.0, 0.0, 0.0))),
                     (5, Pose3((0.0, 0.0, 0.0))),
                     (9, Pose3((0.0, 1.0, 0.0))),
                     (100, Pose3((9.0, 9.0, 9.0)))]
        result = evaluate_trajectory(reference, estimated, alignment="none")
        # Reference order, not estimated order; id 100 ignored.
        self.assertEqual(result.matched_ids, (5, 2, 9, 7))
        origins = [pose.translation for pose in result.aligned_poses]
        self.assertEqual(
            origins,
            [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
        )

    def test_generators_consumed_once(self):
        reference = make_path()
        estimated = make_path()
        result = evaluate_trajectory(
            (record for record in reference),
            (record for record in estimated),
            alignment="none",
        )
        self.assertEqual(len(result.matched_ids), 6)

    def test_inputs_not_modified(self):
        reference = make_path()
        estimated = make_path()
        reference_snapshot = list(reference)
        estimated_snapshot = list(estimated)
        evaluate_trajectory(reference, estimated, alignment="sim3")
        self.assertEqual(reference, reference_snapshot)
        self.assertEqual(estimated, estimated_snapshot)


class RelativeErrorTest(unittest.TestCase):
    def test_known_relative_rotation_error(self):
        reference = [
            (0, Pose3((0.0, 0.0, 0.0))),
            (1, Pose3((1.0, 0.0, 0.0))),
            (2, Pose3((2.0, 0.0, 0.0))),
            (3, Pose3((3.0, 0.0, 0.0))),
        ]
        estimated = [
            (0, Pose3((0.0, 0.0, 0.0))),
            (1, Pose3((1.0, 0.0, 0.0), rotation_z(0.2).quaternion)),
            (2, Pose3((2.0, 0.0, 0.0), rotation_z(0.4).quaternion)),
            (3, Pose3((3.0, 0.0, 0.0), rotation_z(0.6).quaternion)),
        ]
        result = evaluate_trajectory(reference, estimated, alignment="none")
        self.assertEqual(len(result.relative_rotation_errors), 3)
        for error in result.relative_rotation_errors:
            self.assertAlmostEqual(error, 0.2, delta=1e-12)
        self.assertAlmostEqual(result.relative_rotation_rmse, 0.2, delta=1e-12)
        # The estimated relative translation is rotz(-0.2*i) applied to the
        # (1, 0, 0) step, so the error transform's translation has norm
        # 2*|sin(0.1*i)| for the pair starting at frame i.
        expected = [0.0, 2.0 * math.sin(0.1), 2.0 * math.sin(0.2)]
        for error, wanted in zip(result.relative_translation_errors, expected):
            self.assertAlmostEqual(error, wanted, delta=1e-12)

    def test_delta_selects_frame_interval(self):
        reference = [(i, Pose3((float(i), 0.0, 0.0))) for i in range(5)]
        estimated = [(i, Pose3((float(i), 0.1 * i, 0.0))) for i in range(5)]
        one = evaluate_trajectory(reference, estimated, alignment="none", delta=1)
        two = evaluate_trajectory(reference, estimated, alignment="none", delta=2)
        self.assertEqual(len(one.relative_translation_errors), 4)
        self.assertEqual(len(two.relative_translation_errors), 3)
        # Each delta-1 step drifts 0.1 in y; delta-2 steps drift 0.2.
        for error in one.relative_translation_errors:
            self.assertAlmostEqual(error, 0.1, delta=1e-12)
        for error in two.relative_translation_errors:
            self.assertAlmostEqual(error, 0.2, delta=1e-12)

    def test_rmse_matches_public_error_sequences(self):
        reference = make_path()
        estimated = [
            (record_id, pose.compose(Pose3((0.05 * k, -0.02 * k, 0.01 * k))))
            for k, (record_id, pose) in enumerate(reference)
        ]
        result = evaluate_trajectory(reference, estimated, alignment="none", delta=2)
        self.assertAlmostEqual(
            result.translation_rmse, rmse(result.translation_errors), places=15
        )
        self.assertAlmostEqual(
            result.rotation_rmse, rmse(result.rotation_errors), places=15
        )
        self.assertAlmostEqual(
            result.relative_translation_rmse,
            rmse(result.relative_translation_errors),
            places=15,
        )
        self.assertAlmostEqual(
            result.relative_rotation_rmse,
            rmse(result.relative_rotation_errors),
            places=15,
        )


class ImmutabilityTest(unittest.TestCase):
    def test_result_is_immutable(self):
        result = evaluate_trajectory(make_path(), make_path(), alignment="none")
        self.assertIsInstance(result.matched_ids, tuple)
        self.assertIsInstance(result.aligned_poses, tuple)
        self.assertIsInstance(result.translation_errors, tuple)
        self.assertIsInstance(result.rotation_errors, tuple)
        self.assertIsInstance(result.relative_translation_errors, tuple)
        self.assertIsInstance(result.relative_rotation_errors, tuple)
        with self.assertRaises(AttributeError):
            result.scale = 2.0
        with self.assertRaises(AttributeError):
            result.matched_ids = ()

    def test_deterministic_repeated_evaluation(self):
        reference = make_path()
        estimated = make_path()
        first = evaluate_trajectory(reference, estimated, alignment="sim3")
        second = evaluate_trajectory(reference, estimated, alignment="sim3")
        self.assertEqual(first, second)


class ValidationTest(unittest.TestCase):
    def test_non_iterable_trajectory(self):
        with self.assertRaises(TypeError):
            evaluate_trajectory(42, make_path())
        with self.assertRaises(TypeError):
            evaluate_trajectory(make_path(), None)
        with self.assertRaises(TypeError):
            evaluate_trajectory("not a trajectory", make_path())

    def test_record_not_a_pair(self):
        good = make_path()
        with self.assertRaises(TypeError):
            evaluate_trajectory(good, [(0, Pose3(), "extra")] + good[1:])
        with self.assertRaises(TypeError):
            evaluate_trajectory(good, [(0,)] + good[1:])
        with self.assertRaises(TypeError):
            evaluate_trajectory(good, [42] + good[1:])

    def test_id_must_be_non_boolean_integer(self):
        good = make_path()
        with self.assertRaises(TypeError):
            evaluate_trajectory(good, [(True, Pose3())] + good[1:])
        with self.assertRaises(TypeError):
            evaluate_trajectory(good, [(1.5, Pose3())] + good[1:])
        with self.assertRaises(TypeError):
            evaluate_trajectory(good, [("1", Pose3())] + good[1:])

    def test_pose_must_be_pose3(self):
        good = make_path()
        with self.assertRaises(TypeError):
            evaluate_trajectory(good, [(0, "not a pose")] + good[1:])
        with self.assertRaises(TypeError):
            evaluate_trajectory(good, [(0, ((1, 0, 0), (1, 0, 0, 0)))] + good[1:])

    def test_duplicate_ids_raise(self):
        with self.assertRaises(ValueError):
            evaluate_trajectory(
                [(1, Pose3()), (1, Pose3()), (2, Pose3())], make_path()
            )
        with self.assertRaises(ValueError):
            evaluate_trajectory(
                make_path(), [(1, Pose3()), (1, Pose3()), (2, Pose3())]
            )

    def test_alignment_validation(self):
        with self.assertRaises(TypeError):
            evaluate_trajectory(make_path(), make_path(), alignment=7)
        with self.assertRaises(TypeError):
            evaluate_trajectory(make_path(), make_path(), alignment=None)
        with self.assertRaises(ValueError):
            evaluate_trajectory(make_path(), make_path(), alignment="SE3")
        with self.assertRaises(ValueError):
            evaluate_trajectory(make_path(), make_path(), alignment="rigid")

    def test_delta_validation(self):
        with self.assertRaises(TypeError):
            evaluate_trajectory(make_path(), make_path(), delta=True)
        with self.assertRaises(TypeError):
            evaluate_trajectory(make_path(), make_path(), delta=1.5)
        with self.assertRaises(TypeError):
            evaluate_trajectory(make_path(), make_path(), delta="1")
        with self.assertRaises(ValueError):
            evaluate_trajectory(make_path(), make_path(), delta=0)
        with self.assertRaises(ValueError):
            evaluate_trajectory(make_path(), make_path(), delta=-2)

    def test_too_few_common_ids(self):
        reference = [(1, Pose3()), (2, Pose3()), (3, Pose3())]
        estimated = [(1, Pose3()), (2, Pose3()), (4, Pose3())]
        with self.assertRaises(ValueError):
            evaluate_trajectory(reference, estimated)
        with self.assertRaises(ValueError):
            evaluate_trajectory([], [])

    def test_delta_not_smaller_than_match_count(self):
        trajectory = make_path(4)
        with self.assertRaises(ValueError):
            evaluate_trajectory(trajectory, trajectory, delta=4)
        with self.assertRaises(ValueError):
            evaluate_trajectory(trajectory, trajectory, delta=5)
        # delta == count - 1 is accepted.
        result = evaluate_trajectory(trajectory, trajectory, delta=3)
        self.assertEqual(len(result.relative_translation_errors), 1)


class ImportSideEffectTest(unittest.TestCase):
    def test_import_does_not_start_http_service(self):
        code = (
            "import sys\n"
            "import mappilot.evaluation\n"
            "assert 'mappilot.server' not in sys.modules\n"
            "assert 'mappilot.service' not in sys.modules\n"
        )
        subprocess.run([sys.executable, "-c", code], check=True)


if __name__ == "__main__":
    unittest.main()
