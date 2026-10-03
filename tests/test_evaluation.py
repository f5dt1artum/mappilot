import math
import unittest

from mappilot.evaluation import TrajectoryEvaluation, evaluate_trajectory
from mappilot.geometry import Pose3


def rotation_z(angle):
    return Pose3(
        (0.0, 0.0, 0.0),
        (math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0)),
    )


def make_reference(count=6):
    """A non-degenerate (non-collinear, non-coplanar) reference trajectory."""
    poses = []
    for index in range(count):
        translation = (
            float(index),
            float((index * index) % 5) * 0.7,
            float((index % 3) - 1) * 0.9,
        )
        poses.append((index, Pose3(translation, (1.0, 0.0, 0.0, 0.0))))
    return poses


def pose_close(test_case, actual, expected, tol=1e-9):
    for a, e in zip(actual.translation, expected.translation):
        test_case.assertAlmostEqual(a, e, delta=tol)
    qa = actual.quaternion
    qe = expected.quaternion
    test_case.assertTrue(
        all(abs(a - e) <= tol for a, e in zip(qa, qe))
        or all(abs(a + e) <= tol for a, e in zip(qa, qe))
    )


class NoAlignmentTest(unittest.TestCase):
    def test_identity_alignment_and_scale_one(self):
        reference = make_reference()
        estimated = [(i, pose) for i, pose in reference]
        result = evaluate_trajectory(reference, estimated, alignment="none")
        self.assertIsInstance(result, TrajectoryEvaluation)
        pose_close(self, result.alignment_pose, Pose3.identity())
        self.assertEqual(result.scale, 1.0)
        self.assertEqual(result.matched_ids, tuple(range(6)))
        self.assertEqual(result.aligned_poses, tuple(pose for _, pose in estimated))
        self.assertTrue(all(e == 0.0 for e in result.translation_errors))
        self.assertTrue(all(e == 0.0 for e in result.rotation_errors))
        self.assertEqual(result.translation_rmse, 0.0)
        self.assertEqual(result.rotation_rmse, 0.0)

    def test_errors_computed_without_alignment(self):
        reference = make_reference(4)
        estimated = [
            (i, Pose3((x + 3.0, y, z), q)) for i, (x, y, z), q in
            [(i, p.translation, p.quaternion) for i, p in reference]
        ]
        result = evaluate_trajectory(reference, estimated, alignment="none")
        self.assertTrue(
            all(abs(e - 3.0) < 1e-12 for e in result.translation_errors)
        )
        self.assertAlmostEqual(result.translation_rmse, 3.0, delta=1e-12)
        self.assertTrue(all(e == 0.0 for e in result.rotation_errors))

    def test_rotation_error_is_minimal_geodesic_angle(self):
        reference = make_reference(4)
        angle = 0.5
        turn = rotation_z(angle)
        estimated = [(i, turn.compose(p)) for i, p in reference]
        result = evaluate_trajectory(reference, estimated, alignment="none")
        self.assertTrue(
            all(abs(e - angle) < 1e-12 for e in result.rotation_errors)
        )
        self.assertAlmostEqual(result.rotation_rmse, angle, delta=1e-12)

    def test_degenerate_positions_do_not_fail(self):
        reference = [(i, Pose3((float(i), 0.0, 0.0))) for i in range(4)]
        estimated = [(i, Pose3((float(i), 1.0, 0.0))) for i in range(4)]
        result = evaluate_trajectory(reference, estimated, alignment="none")
        self.assertTrue(
            all(abs(e - 1.0) < 1e-12 for e in result.translation_errors)
        )


class Se3AlignmentTest(unittest.TestCase):
    def test_recovers_known_rigid_transform(self):
        reference = make_reference()
        alignment = Pose3((1.0, -2.0, 0.5), (math.cos(0.3), 0.0, 0.0, math.sin(0.3)))
        estimated = [(i, alignment.inverse().compose(p)) for i, p in reference]
        result = evaluate_trajectory(reference, estimated)
        pose_close(self, result.alignment_pose, alignment)
        self.assertEqual(result.scale, 1.0)
        self.assertTrue(all(e < 1e-9 for e in result.translation_errors))
        self.assertTrue(all(e < 1e-9 for e in result.rotation_errors))
        self.assertLess(result.translation_rmse, 1e-9)
        self.assertLess(result.rotation_rmse, 1e-9)

    def test_aligned_poses_left_compose_alignment(self):
        reference = make_reference()
        alignment = Pose3((2.0, 1.0, -1.0), (math.cos(0.2), 0.0, math.sin(0.2), 0.0))
        estimated = [(i, alignment.inverse().compose(p)) for i, p in reference]
        result = evaluate_trajectory(reference, estimated, alignment="se3")
        for aligned, (_, est_pose) in zip(result.aligned_poses, estimated):
            expected = result.alignment_pose.compose(est_pose)
            pose_close(self, aligned, expected)

    def test_collinear_positions_raise(self):
        reference = [(i, Pose3((float(i), 0.0, 0.0))) for i in range(4)]
        estimated = [(i, Pose3((float(i), 1.0, 0.0))) for i in range(4)]
        with self.assertRaises(ValueError):
            evaluate_trajectory(reference, estimated, alignment="se3")


class Sim3AlignmentTest(unittest.TestCase):
    def test_recovers_scale_rotation_and_translation(self):
        reference = make_reference()
        scale = 2.5
        alignment = Pose3((1.0, 2.0, 3.0), (math.cos(0.4), 0.0, 0.0, math.sin(0.4)))
        estimated = []
        for i, pose in reference:
            moved = alignment.transform_point(pose.translation)
            scaled = tuple(scale * c for c in moved)
            estimated.append((i, Pose3(scaled, pose.quaternion)))
        result = evaluate_trajectory(reference, estimated, alignment="sim3")
        self.assertAlmostEqual(result.scale, 1.0 / scale, delta=1e-9)
        self.assertTrue(all(e < 1e-8 for e in result.translation_errors))
        self.assertLess(result.translation_rmse, 1e-8)

    def test_scale_acts_on_translation_only(self):
        reference = make_reference()
        scale = 3.0
        alignment = Pose3((0.5, -1.0, 2.0), (math.cos(0.1), 0.0, math.sin(0.1), 0.0))
        rot = rotation_z(0.7)
        estimated = []
        for i, pose in reference:
            moved = alignment.transform_point(pose.translation)
            scaled = tuple(scale * c for c in moved)
            estimated.append((i, Pose3(scaled, rot.compose(pose).quaternion)))
        result = evaluate_trajectory(reference, estimated, alignment="sim3")
        # Aligned rotation = alignment rotation * estimated rotation, unscaled.
        for aligned, (_, est_pose) in zip(result.aligned_poses, estimated):
            expected_q = result.alignment_pose.compose(est_pose).quaternion
            qa, qe = aligned.quaternion, expected_q
            self.assertTrue(
                all(abs(a - e) <= 1e-9 for a, e in zip(qa, qe))
                or all(abs(a + e) <= 1e-9 for a, e in zip(qa, qe))
            )
        # Aligned translation = scale * (R t + t_align) + (1 - scale) * t_align.
        at = result.alignment_pose.translation
        for aligned, (_, est_pose) in zip(result.aligned_poses, estimated):
            moved = result.alignment_pose.transform_point(est_pose.translation)
            for k in range(3):
                expected = result.scale * moved[k] + (1.0 - result.scale) * at[k]
                self.assertAlmostEqual(aligned.translation[k], expected, delta=1e-9)

    def test_collinear_positions_raise(self):
        reference = [(i, Pose3((float(i), 0.0, 0.0))) for i in range(4)]
        estimated = [(i, Pose3((2.0 * i, 1.0, 0.0))) for i in range(4)]
        with self.assertRaises(ValueError):
            evaluate_trajectory(reference, estimated, alignment="sim3")


class MatchingTest(unittest.TestCase):
    def test_intersection_in_reference_order_ignores_unmatched(self):
        reference = make_reference(5)
        # Shuffled estimated ids, one missing (2), one extra (99).
        estimated = [
            (99, Pose3((0.0, 0.0, 0.0))),
            (4, reference[4][1]),
            (1, reference[1][1]),
            (3, reference[3][1]),
            (0, reference[0][1]),
        ]
        result = evaluate_trajectory(reference, estimated, alignment="none")
        self.assertEqual(result.matched_ids, (0, 1, 3, 4))
        self.assertEqual(len(result.aligned_poses), 4)
        self.assertEqual(len(result.translation_errors), 4)
        self.assertEqual(len(result.rotation_errors), 4)

    def test_generator_inputs_consumed_once(self):
        reference = make_reference(4)
        result = evaluate_trajectory(
            (record for record in reference),
            iter([(i, p) for i, p in reference]),
            alignment="none",
        )
        self.assertEqual(result.matched_ids, (0, 1, 2, 3))

    def test_inputs_not_modified(self):
        reference = make_reference(4)
        estimated = [(i, p) for i, p in reference]
        snapshot = list(reference), list(estimated)
        evaluate_trajectory(reference, estimated)
        self.assertEqual(reference, snapshot[0])
        self.assertEqual(estimated, snapshot[1])


class RelativeErrorTest(unittest.TestCase):
    def test_relative_errors_with_default_delta(self):
        reference = make_reference(4)
        estimated = [(i, p) for i, p in reference]
        result = evaluate_trajectory(reference, estimated, alignment="none")
        self.assertEqual(len(result.relative_translation_errors), 3)
        self.assertEqual(len(result.relative_rotation_errors), 3)
        self.assertTrue(all(e == 0.0 for e in result.relative_translation_errors))
        self.assertTrue(all(e == 0.0 for e in result.relative_rotation_errors))
        self.assertEqual(result.relative_translation_rmse, 0.0)
        self.assertEqual(result.relative_rotation_rmse, 0.0)

    def test_delta_selects_frame_interval(self):
        reference = make_reference(6)
        estimated = [(i, p) for i, p in reference]
        result = evaluate_trajectory(reference, estimated, alignment="none", delta=2)
        self.assertEqual(len(result.relative_translation_errors), 4)
        self.assertEqual(len(result.relative_rotation_errors), 4)

    def test_relative_translation_error_magnitude(self):
        # Constant per-step drift of 0.1 in x on the estimate: every relative
        # translation error equals 0.1 regardless of delta.
        reference = [(i, Pose3((float(i), 0.0, 0.0))) for i in range(5)]
        estimated = [
            (i, Pose3((float(i) + 0.1 * i, 0.0, 0.0))) for i in range(5)
        ]
        result = evaluate_trajectory(reference, estimated, alignment="none", delta=2)
        self.assertTrue(
            all(abs(e - 0.2) < 1e-12 for e in result.relative_translation_errors)
        )
        self.assertAlmostEqual(result.relative_translation_rmse, 0.2, delta=1e-12)

    def test_relative_rotation_error(self):
        reference = make_reference(4)
        # Each estimated step adds a 0.1 rad yaw relative to the reference.
        step = rotation_z(0.1)
        estimated = []
        pose = reference[0][1]
        estimated.append((0, pose))
        for i in range(1, 4):
            pose = pose.compose(Pose3((1.0, 0.0, 0.0))).compose(step)
            estimated.append((i, pose))
        result = evaluate_trajectory(
            [(i, Pose3((float(i), 0.0, 0.0))) for i in range(4)],
            estimated,
            alignment="none",
        )
        self.assertTrue(
            all(abs(e - 0.1) < 1e-12 for e in result.relative_rotation_errors)
        )
        self.assertAlmostEqual(result.relative_rotation_rmse, 0.1, delta=1e-12)

    def test_rmse_matches_public_sequences(self):
        reference = make_reference(5)
        estimated = [
            (i, Pose3((p.translation[0] + 0.1 * i, p.translation[1], p.translation[2])))
            for i, p in reference
        ]
        result = evaluate_trajectory(reference, estimated, alignment="none")
        expected_t = math.sqrt(
            sum(e * e for e in result.translation_errors) / len(result.translation_errors)
        )
        expected_r = math.sqrt(
            sum(e * e for e in result.rotation_errors) / len(result.rotation_errors)
        )
        expected_rt = math.sqrt(
            sum(e * e for e in result.relative_translation_errors)
            / len(result.relative_translation_errors)
        )
        expected_rr = math.sqrt(
            sum(e * e for e in result.relative_rotation_errors)
            / len(result.relative_rotation_errors)
        )
        self.assertAlmostEqual(result.translation_rmse, expected_t)
        self.assertAlmostEqual(result.rotation_rmse, expected_r)
        self.assertAlmostEqual(result.relative_translation_rmse, expected_rt)
        self.assertAlmostEqual(result.relative_rotation_rmse, expected_rr)


class ValidationTest(unittest.TestCase):
    def setUp(self):
        self.reference = make_reference(4)
        self.estimated = [(i, p) for i, p in self.reference]

    def test_non_iterable_trajectory(self):
        with self.assertRaises(TypeError):
            evaluate_trajectory(42, self.estimated)
        with self.assertRaises(TypeError):
            evaluate_trajectory(self.reference, None)

    def test_record_not_a_pair(self):
        with self.assertRaises(TypeError):
            evaluate_trajectory([(1, Pose3(), 3)] * 4, self.estimated)
        with self.assertRaises(TypeError):
            evaluate_trajectory(self.reference, [Pose3()] * 4)
        with self.assertRaises(TypeError):
            evaluate_trajectory(["ab"] * 4, self.estimated)

    def test_non_integer_or_boolean_id(self):
        with self.assertRaises(TypeError):
            evaluate_trajectory(
                [(float(i), p) for i, p in self.reference], self.estimated
            )
        with self.assertRaises(TypeError):
            evaluate_trajectory(
                self.reference, [(bool(i), p) for i, p in self.estimated]
            )

    def test_non_pose3(self):
        with self.assertRaises(TypeError):
            evaluate_trajectory(
                [(i, (0.0, 0.0, 0.0)) for i in range(4)], self.estimated
            )

    def test_non_string_alignment(self):
        with self.assertRaises(TypeError):
            evaluate_trajectory(self.reference, self.estimated, alignment=1)

    def test_unsupported_alignment(self):
        with self.assertRaises(ValueError):
            evaluate_trajectory(self.reference, self.estimated, alignment="SE3")
        with self.assertRaises(ValueError):
            evaluate_trajectory(self.reference, self.estimated, alignment="rigid")

    def test_non_integer_or_boolean_delta(self):
        with self.assertRaises(TypeError):
            evaluate_trajectory(self.reference, self.estimated, delta=1.5)
        with self.assertRaises(TypeError):
            evaluate_trajectory(self.reference, self.estimated, delta=True)

    def test_delta_below_one(self):
        with self.assertRaises(ValueError):
            evaluate_trajectory(self.reference, self.estimated, delta=0)

    def test_duplicate_ids(self):
        with self.assertRaises(ValueError):
            evaluate_trajectory(
                [(0, Pose3())] * 2 + [(i, Pose3()) for i in range(1, 4)],
                self.estimated,
            )
        with self.assertRaises(ValueError):
            evaluate_trajectory(
                self.reference,
                [(0, Pose3())] * 2 + [(i, Pose3()) for i in range(1, 4)],
            )

    def test_too_few_common_ids(self):
        estimated = [(i + 10, p) for i, p in self.estimated]
        estimated[0] = (0, estimated[0][1])
        estimated[1] = (1, estimated[1][1])
        with self.assertRaises(ValueError):
            evaluate_trajectory(self.reference, estimated)

    def test_delta_not_smaller_than_match_count(self):
        with self.assertRaises(ValueError):
            evaluate_trajectory(self.reference, self.estimated, delta=4)


class ImmutabilityTest(unittest.TestCase):
    def test_result_sequences_are_tuples(self):
        reference = make_reference(4)
        result = evaluate_trajectory(reference, reference, alignment="none")
        for value in (
            result.matched_ids,
            result.aligned_poses,
            result.translation_errors,
            result.rotation_errors,
            result.relative_translation_errors,
            result.relative_rotation_errors,
        ):
            self.assertIsInstance(value, tuple)
        with self.assertRaises(AttributeError):
            result.scale = 2.0


if __name__ == "__main__":
    unittest.main()
