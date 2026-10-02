"""Tests for mappilot.geometry.Pose3."""

from __future__ import annotations

import math
import unittest

from mappilot import Pose3
from mappilot.geometry import _quaternion_multiply  # type: ignore

I3 = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


def rot_x(angle: float) -> tuple[tuple[float, ...], ...]:
    c, s = math.cos(angle), math.sin(angle)
    return (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, c, -s, 0.0),
        (0.0, s, c, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def mat_mul(a, b):
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4))
        for i in range(4)
    )


class ConstructionTest(unittest.TestCase):
    def test_identity_defaults(self) -> None:
        pose = Pose3()
        self.assertEqual(pose.translation, (0.0, 0.0, 0.0))
        self.assertEqual(pose.quaternion, (1.0, 0.0, 0.0, 0.0))
        self.assertEqual(Pose3.identity().quaternion, (1.0, 0.0, 0.0, 0.0))

    def test_tq_aliases(self) -> None:
        pose = Pose3.from_tq((1, 2, 3), (1, 0, 0, 0))
        self.assertEqual(pose.translation, (1.0, 2.0, 3.0))
        self.assertEqual(
            Pose3.from_translation_quaternion((1, 2, 3), (1, 0, 0, 0)), pose
        )

    def test_non_unit_quaternion_is_normalized(self) -> None:
        pose = Pose3((0, 0, 0), (0.0, 2.0, 0.0, 0.0))
        q = pose.quaternion
        self.assertAlmostEqual(q[0], 0.0, places=12)
        self.assertAlmostEqual(q[1], 1.0, places=12)
        norm = math.sqrt(sum(component * component for component in q))
        self.assertAlmostEqual(norm, 1.0, places=12)

    def test_outputs_are_tuples_and_defensive(self) -> None:
        source_t = [0.1, 0.2, 0.3]
        source_q = [1.0, 0.0, 0.0, 0.0]
        pose = Pose3(source_t, source_q)
        source_t[0] = 99.0
        source_q[1] = 99.0
        self.assertEqual(pose.translation, (0.1, 0.2, 0.3))
        self.assertEqual(pose.quaternion, (1.0, 0.0, 0.0, 0.0))
        self.assertIsInstance(pose.translation, tuple)
        self.assertIsInstance(pose.quaternion, tuple)
        self.assertIsInstance(pose.to_matrix(), tuple)
        self.assertIsInstance(pose.to_matrix()[0], tuple)

    def test_quaternion_sign_canonical(self) -> None:
        q = (0.5, 0.5, 0.5, 0.5)
        neg = tuple(-component for component in q)
        self.assertEqual(Pose3((0, 0, 0), q), Pose3((0, 0, 0), neg))
        self.assertGreaterEqual(Pose3((0, 0, 0), q).quaternion[0], 0.0)

    def test_quaternion_zero_w_first_nonzero_positive(self) -> None:
        # Rotation by pi about y: canonical quaternion (0, 0, 1, 0).
        pose_pos = Pose3((0, 0, 0), (0.0, 0.0, 3.0, 0.0))
        pose_neg = Pose3((0, 0, 0), (0.0, 0.0, -3.0, 0.0))
        self.assertEqual(pose_pos.quaternion, (0.0, 0.0, 1.0, 0.0))
        self.assertEqual(pose_neg.quaternion, (0.0, 0.0, 1.0, 0.0))
        # Rotation by pi about x: (0, 1, 0, 0).
        self.assertEqual(
            Pose3((0, 0, 0), (0.0, -2.0, 0.0, 0.0)).quaternion,
            (0.0, 1.0, 0.0, 0.0),
        )

    def test_matrix_construction_identity(self) -> None:
        self.assertEqual(Pose3.from_matrix(I3), Pose3.identity())

    def test_matrix_construction_roundtrip(self) -> None:
        original = Pose3.exp((0.3, -0.7, 0.5, 1.5, -2.0, 0.25))
        rebuilt = Pose3.from_matrix(original.to_matrix())
        for i in range(4):
            for j in range(4):
                self.assertAlmostEqual(
                    rebuilt.to_matrix()[i][j],
                    original.to_matrix()[i][j],
                    places=9,
                )

    def test_matrix_last_row_checked(self) -> None:
        bad = [list(row) for row in I3]
        bad[3] = [0.0, 0.0, 1e-8, 1.0]
        with self.assertRaises(ValueError):
            Pose3.from_matrix(bad)
        bad2 = [list(row) for row in I3]
        bad2[3][3] = 1.0 + 2e-9
        with self.assertRaises(ValueError):
            Pose3.from_matrix(bad2)

    def test_matrix_rotation_validated(self) -> None:
        # Reflection (det = -1) must be rejected, not repaired.
        reflection = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        with self.assertRaises(ValueError):
            Pose3.from_matrix(reflection)
        non_orthogonal = [
            [1.0 + 2e-8, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        with self.assertRaises(ValueError):
            Pose3.from_matrix(non_orthogonal)

    def test_near_orthogonal_matrix_accepted_at_tolerance(self) -> None:
        near = [
            [1.0, 0.0, 5e-10, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-5e-10, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        pose = Pose3.from_matrix(near)
        self.assertTrue(all(math.isfinite(v) for v in pose.quaternion))


class TypeErrorValidationTest(unittest.TestCase):
    def test_scalars_reject_bool(self) -> None:
        with self.assertRaises(TypeError):
            Pose3((True, 0.0, 0.0), (1, 0, 0, 0))
        with self.assertRaises(TypeError):
            Pose3((0, 0, 0), (1, True, 0, 0))
        with self.assertRaises(TypeError):
            Pose3.exp((0, 0, 0, True, 0, 0))
        with self.assertRaises(TypeError):
            Pose3().transform((0, 0, True))

    def test_scalars_reject_non_real(self) -> None:
        with self.assertRaises(TypeError):
            Pose3(("0", 0, 0), (1, 0, 0, 0))
        with self.assertRaises(TypeError):
            Pose3((0, 0, 0), (1, None, 0, 0))
        with self.assertRaises(TypeError):
            Pose3((1 + 2j, 0, 0), (1, 0, 0, 0))

    def test_wrong_dimensions_are_type_errors(self) -> None:
        with self.assertRaises(TypeError):
            Pose3((0, 0), (1, 0, 0, 0))
        with self.assertRaises(TypeError):
            Pose3((0, 0, 0), (1, 0, 0))
        with self.assertRaises(TypeError):
            Pose3.exp((0, 0, 0, 0, 0))
        with self.assertRaises(TypeError):
            Pose3().transform((0, 0))
        with self.assertRaises(TypeError):
            Pose3.from_matrix([[1, 0, 0, 0]] * 3)
        with self.assertRaises(TypeError):
            Pose3.from_matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])

    def test_non_iterable_and_scalar_inputs(self) -> None:
        with self.assertRaises(TypeError):
            Pose3(7, (1, 0, 0, 0))
        with self.assertRaises(TypeError):
            Pose3.from_matrix(7)
        with self.assertRaises(TypeError):
            Pose3().transform_points(7)
        with self.assertRaises(TypeError):
            Pose3().transform_points([7])

    def test_nan_and_infinity_are_value_errors(self) -> None:
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                Pose3((bad, 0, 0), (1, 0, 0, 0))
            with self.assertRaises(ValueError):
                Pose3((0, 0, 0), (bad, 0, 0, 0))
            with self.assertRaises(ValueError):
                Pose3.exp((bad, 0, 0, 0, 0, 0))

    def test_zero_quaternion_is_value_error(self) -> None:
        with self.assertRaises(ValueError):
            Pose3((0, 0, 0), (0, 0, 0, 0))

    def test_mixed_constructor_arguments_rejected(self) -> None:
        with self.assertRaises(TypeError):
            Pose3((0, 0, 0), (1, 0, 0, 0), matrix=I3)

    def test_compose_type_check(self) -> None:
        with self.assertRaises(TypeError):
            Pose3.identity().compose((1, 0, 0, 0, 0, 0, 0))


class ComposeInverseTransformTest(unittest.TestCase):
    def setUp(self) -> None:
        self.a = Pose3.exp((0.4, -0.2, 0.8, 1.0, -2.0, 0.5))
        self.b = Pose3.exp((-0.6, 0.3, 0.1, 0.0, 1.0, 2.0))

    def test_compose_matches_matrix_product(self) -> None:
        composed = self.a.compose(self.b)
        product = mat_mul(self.a.to_matrix(), self.b.to_matrix())
        for i in range(4):
            for j in range(4):
                self.assertAlmostEqual(
                    composed.to_matrix()[i][j], product[i][j], places=9
                )

    def test_compose_operator(self) -> None:
        self.assertEqual(self.a @ self.b, self.a.compose(self.b))

    def test_compose_order_b_first(self) -> None:
        # Pure translations make the order observable on points.
        a = Pose3((1, 0, 0), (1, 0, 0, 0))
        b = Pose3((0, 1, 0), (1, 0, 0, 0))
        self.assertEqual(a.compose(b).transform((0, 0, 0)), (1, 1, 0))

    def test_inverse_composes_to_identity(self) -> None:
        for pose in (self.a, self.b):
            identity = pose.inverse().compose(pose)
            self.assert_pose_identity(identity)
            identity = pose.compose(pose.inverse())
            self.assert_pose_identity(identity)

    def assert_pose_identity(self, pose: Pose3) -> None:
        for i in range(4):
            for j in range(4):
                self.assertAlmostEqual(
                    pose.to_matrix()[i][j], I3[i][j], places=9
                )

    def test_inverse_matches_matrix_inverse(self) -> None:
        matrix = self.a.to_matrix()
        r = tuple(row[:3] for row in matrix[:3])
        t = tuple(row[3] for row in matrix[:3])
        expected_t = (
            -sum(r[j][0] * t[j] for j in range(3)),
            -sum(r[j][1] * t[j] for j in range(3)),
            -sum(r[j][2] * t[j] for j in range(3)),
        )
        inv = self.a.inverse()
        for i in range(3):
            self.assertAlmostEqual(inv.translation[i], expected_t[i], places=9)

    def test_transform_point(self) -> None:
        pose = Pose3((10, 20, 30), (0, 0, 0, 1))  # 180 deg about z
        point = pose.transform((1, 2, 3))
        self.assertAlmostEqual(point[0], 9.0, places=9)
        self.assertAlmostEqual(point[1], 18.0, places=9)
        self.assertAlmostEqual(point[2], 33.0, places=9)
        self.assertIsInstance(point, tuple)

    def test_transform_consistent_with_compose(self) -> None:
        point = (0.7, -1.3, 2.1)
        moved = self.a.compose(Pose3(point, (1, 0, 0, 0)))
        self.assertTrue(
            all(
                abs(moved.translation[i] - self.a.transform(point)[i]) < 1e-9
                for i in range(3)
            )
        )

    def test_transform_points_order_and_count(self) -> None:
        points = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)]
        result = self.b.transform_points(points)
        self.assertEqual(len(result), 4)
        self.assertIsInstance(result, tuple)
        for given, got in zip(points, result):
            expected = self.b.transform(given)
            for i in range(3):
                self.assertAlmostEqual(got[i], expected[i], places=12)

    def test_transform_points_empty(self) -> None:
        self.assertEqual(self.a.transform_points([]), ())

    def test_transform_points_accepts_generator_and_tuples(self) -> None:
        result = self.a.transform_points((i, i + 1, i + 2) for i in range(3))
        self.assertEqual(len(result), 3)
        self.assertTrue(all(isinstance(item, tuple) for item in result))

    def test_input_mutation_after_transform(self) -> None:
        points = [[1.0, 2.0, 3.0]]
        result = self.a.transform_points(points)
        points[0][0] = 999.0
        self.assertNotEqual(result[0][0], 999.0)


class ExpLogTest(unittest.TestCase):
    def setUp(self) -> None:
        self.pose = Pose3.exp((0.3, -0.7, 0.5, 1.5, -2.0, 0.25))

    def _assert_vec_close(self, a, b, places=9) -> None:
        self.assertEqual(len(a), len(b))
        for av, bv in zip(a, b):
            self.assertAlmostEqual(av, bv, places=places)

    def test_exp_zero_is_identity(self) -> None:
        pose = Pose3.exp((0, 0, 0, 0, 0, 0))
        self.assertEqual(pose, Pose3.identity())

    def test_exp_translation_only(self) -> None:
        pose = Pose3.exp((0, 0, 0, 1, 2, 3))
        self._assert_vec_close(pose.translation, (1, 2, 3))
        self.assertEqual(pose.quaternion, (1, 0, 0, 0))

    def test_exp_pure_rotation_matches_matrix(self) -> None:
        tangent = (0.3, -0.5, 0.9, 0, 0, 0)
        pose = Pose3.exp(tangent)
        theta = math.sqrt(sum(v * v for v in tangent[:3]))
        axis = tuple(v / theta for v in tangent[:3])
        q_expected = (
            math.cos(theta / 2),
            *(axis[i] * math.sin(theta / 2) for i in range(3)),
        )
        self._assert_vec_close(pose.quaternion, q_expected)

    def test_log_zero(self) -> None:
        self.assertEqual(Pose3.identity().log(), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    def test_log_translation_only(self) -> None:
        tangent = Pose3((4, -5, 6), (1, 0, 0, 0)).log()
        self._assert_vec_close(tangent, (0, 0, 0, 4, -5, 6))

    def test_log_angle_in_range_and_order(self) -> None:
        tangent = self.pose.log()
        angle = math.sqrt(sum(v * v for v in tangent[:3]))
        self.assertGreaterEqual(angle, 0.0)
        self.assertLessEqual(angle, math.pi + 1e-12)

    def test_log_exp_roundtrip_rotation_vectors(self) -> None:
        # angle = norm(w) in [0, pi], arbitrary axes.
        cases = [
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (-0.3, 0.6, 0.2),
            (0.9, -0.4, 0.5),
        ]
        translations = [(0, 0, 0), (1, -2, 0.5), (-3, 0.1, 2.2)]
        for axis in cases:
            norm = math.sqrt(sum(v * v for v in axis))
            unit = tuple(v / norm for v in axis)
            for fraction in (0.05, 0.5, 1.0 - 1e-4, 1.0):
                angle = fraction * math.pi
                w = tuple(angle * component for component in unit)
                for velocity in translations:
                    tangent = (*w, *velocity)
                    with self.subTest(tangent=tangent):
                        pose = Pose3.exp(tangent)
                        recovered = Pose3.exp(pose.log())
                        for i in range(4):
                            for j in range(4):
                                self.assertAlmostEqual(
                                    recovered.to_matrix()[i][j],
                                    pose.to_matrix()[i][j],
                                    places=8,
                                )

    def test_exp_log_near_zero_continuous(self) -> None:
        epsilon = 1e-8
        tiny = Pose3.exp((epsilon, -epsilon, 0.0, 1.0, 2.0, 3.0))
        tangent = tiny.log()
        self._assert_vec_close(tangent, (epsilon, -epsilon, 0.0, 1.0, 2.0, 3.0), places=6)
        # No spurious NaN at the zero/nonzero boundary.
        for scale in (0.0, 1e-12, 1e-9, 1e-6):
            pose = Pose3.exp((scale, 0.0, 0.0, 0.5, -0.5, 0.25))
            self.assertTrue(all(math.isfinite(v) for v in pose.log()))

    def test_log_near_pi_axis_stable(self) -> None:
        axis = (0.0, 1.0, 0.0)  # w == 0 case: first nonzero comp of axis is y
        angle = math.pi - 1e-7
        w = tuple(angle * v for v in axis)
        pose = Pose3.exp((*w, 0.7, -0.7, 0.7))
        tangent = pose.log()
        self._assert_vec_close(tangent[:3], w, places=5)
        # The y component must be positive deterministically (no sign flip).
        self.assertGreater(tangent[1], 0.0)
        self.assertAlmostEqual(tangent[0], 0.0, places=8)
        # Repeated calls are bit-stable.
        self.assertEqual(pose.log(), pose.log())

    def test_log_near_pi_axis_x(self) -> None:
        angle = math.pi - 1e-6
        pose = Pose3.exp((angle, 0.0, 0.0, 1.0, 1.0, 1.0))
        tangent = pose.log()
        self.assertGreater(tangent[0], 0.0)
        self._assert_vec_close(tangent[:3], (angle, 0.0, 0.0), places=5)

    def test_exp_log_full_roundtrip_randomish(self) -> None:
        tangent = (0.83, -1.2, 0.44, 2.1, -0.6, 1.3)
        pose = Pose3.exp(tangent)
        again = Pose3.exp(pose.log())
        for i in range(4):
            for j in range(4):
                self.assertAlmostEqual(
                    again.to_matrix()[i][j], pose.to_matrix()[i][j], places=9
                )

    def test_left_convention_checked_against_formula(self) -> None:
        # For a rotation-only left perturbation, exp(w, 0).rotation must equal
        # the Rodrigues rotation generated by w directly.
        w = (0.25, -0.4, 0.6)
        theta = math.sqrt(sum(v * v for v in w))
        k = tuple(v / theta for v in w)
        c, s = math.cos(theta), math.sin(theta)
        r = Pose3.exp((*w, 0, 0, 0)).to_matrix()

        def skew(vector):
            return (
                (0.0, -vector[2], vector[1]),
                (vector[2], 0.0, -vector[0]),
                (-vector[1], vector[0], 0.0),
            )

        for i in range(3):
            for j in range(3):
                rod = (
                    c * (1.0 if i == j else 0.0)
                    + (1 - c) * k[i] * k[j]
                    + s * skew(k)[i][j]
                )
                self.assertAlmostEqual(r[i][j], rod, places=9)


class QuaternionMatrixRoundTripTest(unittest.TestCase):
    def test_quaternion_matrix_quaternion(self) -> None:
        for w, x, y, z in (
            (1, 0, 0, 0),
            (0, 0, 1, 0),
            (0.5, 0.5, 0.5, 0.5),
            (0.7071, 0.0, 0.7071, 0.0),
        ):
            pose = Pose3((0, 0, 0), (w, x, y, z))
            rebuilt = Pose3.from_matrix(pose.to_matrix())
            for a, b in zip(pose.quaternion, rebuilt.quaternion):
                self.assertAlmostEqual(a, b, places=9)

    def test_all_results_finite(self) -> None:
        pose = Pose3.exp((1.2, -0.9, 0.3, 4.0, -5.0, 6.0))
        self.assertTrue(all(math.isfinite(v) for v in pose.quaternion))
        self.assertTrue(all(math.isfinite(v) for v in pose.translation))
        self.assertTrue(
            all(math.isfinite(x) for row in pose.to_matrix() for x in row)
        )
        self.assertTrue(all(math.isfinite(v) for v in pose.log()))
        self.assertTrue(
            all(math.isfinite(x) for p in pose.transform_points([(1, 2, 3)]) for x in p)
        )

    def test_internal_product_helper_orientation(self) -> None:
        # Sanity: Hamilton product qx * qy = qz.
        qx = (0.0, 1.0, 0.0, 0.0)
        qy = (0.0, 0.0, 1.0, 0.0)
        qz = (0.0, 0.0, 0.0, 1.0)
        self.assertEqual(
            tuple(round(v, 12) for v in _quaternion_multiply(qx, qy)), qz
        )


class ImmutabilityTest(unittest.TestCase):
    def test_slots_no_dict(self) -> None:
        pose = Pose3((1, 2, 3), (1, 0, 0, 0))
        with self.assertRaises(AttributeError):
            pose.arbitrary = 1  # type: ignore[attr-defined]

    def test_hashable_and_equal(self) -> None:
        a = Pose3((1, 2, 3), (0, 1, 0, 0))
        b = Pose3((1, 2, 3), (0, -1, 0, 0))
        self.assertEqual(hash(a), hash(b))
        self.assertEqual({a, b} & {b}, {a})


if __name__ == "__main__":
    unittest.main()
