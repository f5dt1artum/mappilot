import math
import unittest

from mappilot.geometry import Pose3


def assert_tuple_close(test_case, actual, expected, tol=1e-9):
    test_case.assertEqual(len(actual), len(expected))
    for a, e in zip(actual, expected):
        test_case.assertAlmostEqual(a, e, delta=tol)


class ConstructionTest(unittest.TestCase):
    def test_default_is_identity(self):
        pose = Pose3()
        self.assertEqual(pose.translation, (0.0, 0.0, 0.0))
        self.assertEqual(pose.quaternion, (1.0, 0.0, 0.0, 0.0))

    def test_non_unit_quaternion_is_normalized(self):
        pose = Pose3(quaternion=(2.0, 0.0, 0.0, 0.0))
        self.assertEqual(pose.quaternion, (1.0, 0.0, 0.0, 0.0))

    def test_quaternion_canonical_sign_positive_w(self):
        pose = Pose3(quaternion=(-0.5, -0.5, -0.5, -0.5))
        self.assertEqual(pose.quaternion, (0.5, 0.5, 0.5, 0.5))

    def test_quaternion_canonical_sign_zero_w(self):
        pose = Pose3(quaternion=(0.0, -1.0, 0.0, 0.0))
        self.assertEqual(pose.quaternion, (0.0, 1.0, 0.0, 0.0))
        pose = Pose3(quaternion=(0.0, 0.0, 0.0, -3.0))
        self.assertEqual(pose.quaternion, (0.0, 0.0, 0.0, 1.0))

    def test_returns_are_tuples(self):
        pose = Pose3((1, 2, 3), (1, 0, 0, 0))
        self.assertIsInstance(pose.translation, tuple)
        self.assertIsInstance(pose.quaternion, tuple)
        self.assertIsInstance(pose.matrix, tuple)
        for row in pose.matrix:
            self.assertIsInstance(row, tuple)

    def test_input_mutation_does_not_affect_pose(self):
        translation = [1.0, 2.0, 3.0]
        quaternion = [1.0, 0.0, 0.0, 0.0]
        pose = Pose3(translation, quaternion)
        translation[0] = 99.0
        quaternion[0] = 0.0
        quaternion[1] = 1.0
        self.assertEqual(pose.translation, (1.0, 2.0, 3.0))
        self.assertEqual(pose.quaternion, (1.0, 0.0, 0.0, 0.0))

    def test_accepts_int_scalars(self):
        pose = Pose3((1, 2, 3), (1, 0, 0, 0))
        self.assertEqual(pose.translation, (1.0, 2.0, 3.0))


class ValidationTest(unittest.TestCase):
    def test_bool_scalars_rejected(self):
        with self.assertRaises(TypeError):
            Pose3((True, 0.0, 0.0))
        with self.assertRaises(TypeError):
            Pose3(quaternion=(1.0, 0.0, 0.0, False))

    def test_wrong_dimensions_raise_type_error(self):
        with self.assertRaises(TypeError):
            Pose3((1.0, 2.0))
        with self.assertRaises(TypeError):
            Pose3(quaternion=(1.0, 0.0, 0.0))
        with self.assertRaises(TypeError):
            Pose3(1.5)
        with self.assertRaises(TypeError):
            Pose3("abc")

    def test_non_real_elements_raise_type_error(self):
        with self.assertRaises(TypeError):
            Pose3((1.0, "x", 3.0))
        with self.assertRaises(TypeError):
            Pose3(quaternion=(1.0, 0.0, 0.0, None))

    def test_non_finite_values_raise_value_error(self):
        for bad in (math.nan, math.inf, -math.inf):
            with self.assertRaises(ValueError):
                Pose3((bad, 0.0, 0.0))
            with self.assertRaises(ValueError):
                Pose3(quaternion=(1.0, bad, 0.0, 0.0))

    def test_zero_length_quaternion_raises_value_error(self):
        with self.assertRaises(ValueError):
            Pose3(quaternion=(0.0, 0.0, 0.0, 0.0))

    def test_matrix_shape_errors_raise_type_error(self):
        with self.assertRaises(TypeError):
            Pose3.from_matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        with self.assertRaises(TypeError):
            Pose3.from_matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])
        with self.assertRaises(TypeError):
            Pose3.from_matrix("not a matrix")
        with self.assertRaises(TypeError):
            Pose3.from_matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, True]])

    def test_matrix_last_row_raises_value_error(self):
        with self.assertRaises(ValueError):
            Pose3.from_matrix(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1e-8, 1]]
            )
        with self.assertRaises(ValueError):
            Pose3.from_matrix(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 2]]
            )

    def test_reflection_matrix_rejected(self):
        with self.assertRaises(ValueError):
            Pose3.from_matrix(
                [[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
            )

    def test_non_orthogonal_matrix_rejected(self):
        with self.assertRaises(ValueError):
            Pose3.from_matrix(
                [[2, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
            )
        with self.assertRaises(ValueError):
            Pose3.from_matrix(
                [[1, 1e-8, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
            )

    def test_matrix_with_nan_raises_value_error(self):
        with self.assertRaises(ValueError):
            Pose3.from_matrix(
                [[1, 0, 0, math.nan], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
            )


class MatrixRoundTripTest(unittest.TestCase):
    def test_matrix_property(self):
        angle = math.pi / 2.0
        pose = Pose3((1.0, 2.0, 3.0), (math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0)))
        m = pose.matrix
        assert_tuple_close(self, m[0], (0.0, -1.0, 0.0, 1.0))
        assert_tuple_close(self, m[1], (1.0, 0.0, 0.0, 2.0))
        assert_tuple_close(self, m[2], (0.0, 0.0, 1.0, 3.0))
        self.assertEqual(m[3], (0.0, 0.0, 0.0, 1.0))

    def test_quaternion_matrix_quaternion_round_trip(self):
        quats = [
            (1.0, 0.0, 0.0, 0.0),
            (0.5, 0.5, 0.5, 0.5),
            (0.0, 1.0, 0.0, 0.0),
            (math.cos(0.3), 0.0, math.sin(0.3), 0.0),
            (0.36, 0.48, 0.6, 0.6),
        ]
        for q in quats:
            with self.subTest(q=q):
                pose = Pose3((1.0, -2.0, 0.5), q)
                back = Pose3.from_matrix(pose.matrix)
                assert_tuple_close(self, back.quaternion, pose.quaternion)
                assert_tuple_close(self, back.translation, pose.translation)

    def test_matrix_quaternion_matrix_round_trip(self):
        pose = Pose3((3.0, -1.0, 2.0), (0.36, 0.48, 0.6, 0.6))
        m = pose.matrix
        back = Pose3.from_matrix(m).matrix
        for row_a, row_b in zip(m, back):
            assert_tuple_close(self, row_a, row_b)


class GroupOperationTest(unittest.TestCase):
    def test_compose_applies_other_first(self):
        move = Pose3((1.0, 0.0, 0.0))
        half = math.sqrt(0.5)
        turn_z = Pose3(quaternion=(half, 0.0, 0.0, half))
        combined = move.compose(turn_z)
        # turn_z first: (1,0,0) -> (0,1,0); then move: -> (1,1,0)
        assert_tuple_close(self, combined.transform_point((1.0, 0.0, 0.0)), (1.0, 1.0, 0.0))

    def test_compose_matches_matrix_product(self):
        a = Pose3((1.0, 2.0, 3.0), (0.36, 0.48, 0.6, 0.6))
        b = Pose3((-2.0, 0.5, 1.0), (0.6, -0.48, 0.36, 0.6))
        composed = a.compose(b)
        ma, mb = a.matrix, b.matrix
        expected = tuple(
            tuple(sum(ma[i][k] * mb[k][j] for k in range(4)) for j in range(4))
            for i in range(4)
        )
        for row_a, row_e in zip(composed.matrix, expected):
            assert_tuple_close(self, row_a, row_e)

    def test_inverse_yields_identity(self):
        pose = Pose3((1.0, -2.0, 0.5), (0.36, 0.48, 0.6, 0.6))
        for combined in (pose.compose(pose.inverse()), pose.inverse().compose(pose)):
            assert_tuple_close(self, combined.translation, (0.0, 0.0, 0.0))
            assert_tuple_close(self, combined.quaternion, (1.0, 0.0, 0.0, 0.0))

    def test_inverse_undoes_point_transform(self):
        pose = Pose3((1.0, -2.0, 0.5), (0.36, 0.48, 0.6, 0.6))
        point = (0.3, -1.2, 2.4)
        assert_tuple_close(self, pose.inverse().transform_point(pose.transform_point(point)), point)

    def test_compose_rejects_non_pose(self):
        with self.assertRaises(TypeError):
            Pose3().compose((1.0, 2.0, 3.0))


class PointTransformTest(unittest.TestCase):
    def test_transform_point(self):
        half = math.sqrt(0.5)
        pose = Pose3((1.0, 0.0, 0.0), (half, 0.0, 0.0, half))
        assert_tuple_close(self, pose.transform_point((1.0, 0.0, 0.0)), (1.0, 1.0, 0.0))

    def test_transform_points_preserves_order_and_count(self):
        pose = Pose3((10.0, 0.0, 0.0))
        points = [(0.0, 0.0, 0.0), (1.0, 2.0, 3.0), (-1.0, -2.0, -3.0)]
        result = pose.transform_points(points)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)
        assert_tuple_close(self, result[0], (10.0, 0.0, 0.0))
        assert_tuple_close(self, result[1], (11.0, 2.0, 3.0))
        assert_tuple_close(self, result[2], (9.0, -2.0, -3.0))

    def test_transform_points_empty(self):
        self.assertEqual(Pose3().transform_points([]), ())

    def test_transform_points_validates_elements(self):
        with self.assertRaises(TypeError):
            Pose3().transform_points([(1.0, 2.0)])
        with self.assertRaises(ValueError):
            Pose3().transform_points([(1.0, math.nan, 3.0)])


class ExpLogTest(unittest.TestCase):
    def test_exp_of_zero_is_identity(self):
        pose = Pose3.exp((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        self.assertEqual(pose, Pose3())

    def test_log_of_identity_is_zero(self):
        self.assertEqual(Pose3().log(), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    def test_pure_translation(self):
        pose = Pose3.exp((0.0, 0.0, 0.0, 1.0, 2.0, 3.0))
        assert_tuple_close(self, pose.translation, (1.0, 2.0, 3.0))
        assert_tuple_close(self, pose.log(), (0.0, 0.0, 0.0, 1.0, 2.0, 3.0))

    def test_pure_rotation(self):
        pose = Pose3.exp((0.0, 0.0, 1.0, 0.0, 0.0, 0.0))
        assert_tuple_close(self, pose.log(), (0.0, 0.0, 1.0, 0.0, 0.0, 0.0))
        assert_tuple_close(self, pose.transform_point((1.0, 0.0, 0.0)),
                           (math.cos(1.0), math.sin(1.0), 0.0))

    def test_exp_log_round_trip(self):
        tangents = [
            (0.1, -0.2, 0.3, 1.0, -2.0, 0.5),
            (1.0, 0.5, -0.25, -3.0, 2.0, 1.0),
            (2.5, 1.0, 0.5, 0.1, 0.2, 0.3),
            (1e-6, -2e-6, 3e-6, 1e-6, 1.0, -1.0),
            (1e-9, 0.0, 0.0, 1.0, 2.0, 3.0),
            (math.pi - 1e-6, 0.0, 0.0, 1.0, 2.0, 3.0),
            (0.0, math.pi, 0.0, 0.5, -0.5, 2.0),
        ]
        for xi in tangents:
            with self.subTest(xi=xi):
                assert_tuple_close(self, Pose3.exp(xi).log(), xi)

    def test_log_exp_round_trip(self):
        poses = [
            Pose3(),
            Pose3((1.0, 2.0, 3.0), (0.36, 0.48, 0.6, 0.6)),
            Pose3((-0.5, 0.25, 4.0), (0.0, 0.0, 1.0, 0.0)),
            Pose3((1e-3, 0.0, 0.0), (1.0, 1e-9, 0.0, 0.0)),
        ]
        for pose in poses:
            with self.subTest(pose=pose):
                back = Pose3.exp(pose.log())
                assert_tuple_close(self, back.translation, pose.translation)
                assert_tuple_close(self, back.quaternion, pose.quaternion)

    def test_rotation_angle_bounded_by_pi(self):
        # A rotation vector longer than pi maps back to an angle in [0, pi].
        xi = (0.0, 0.0, 4.0, 0.0, 0.0, 0.0)
        logged = Pose3.exp(xi).log()
        angle = math.hypot(*logged[:3])
        self.assertLessEqual(angle, math.pi + 1e-12)
        self.assertGreaterEqual(angle, 0.0)

    def test_axis_stable_near_pi(self):
        # Negative-axis pi rotation canonicalizes to the positive axis.
        logged = Pose3.exp((-math.pi, 0.0, 0.0, 0.0, 0.0, 0.0)).log()
        assert_tuple_close(self, logged[:3], (math.pi, 0.0, 0.0))
        logged = Pose3.exp((0.0, 0.0, -math.pi, 1.0, 2.0, 3.0)).log()
        assert_tuple_close(self, logged[:3], (0.0, 0.0, math.pi))

    def test_continuity_near_zero_angle(self):
        base = Pose3.exp((1e-7, 0.0, 0.0, 0.0, 0.0, 0.0)).log()
        plus = Pose3.exp((1.1e-7, 0.0, 0.0, 0.0, 0.0, 0.0)).log()
        assert_tuple_close(self, plus, tuple(v * 1.1 for v in base), tol=1e-12)

    def test_exp_validates_input(self):
        with self.assertRaises(TypeError):
            Pose3.exp((0.0, 0.0, 0.0, 0.0, 0.0))
        with self.assertRaises(TypeError):
            Pose3.exp((0.0, 0.0, 0.0, 0.0, 0.0, True))
        with self.assertRaises(ValueError):
            Pose3.exp((0.0, 0.0, math.inf, 0.0, 0.0, 0.0))

    def test_results_are_finite(self):
        pose = Pose3.exp((0.7, -1.2, 2.6, 3.0, -4.0, 5.0))
        values = pose.translation + pose.quaternion + pose.log()
        values += tuple(v for row in pose.matrix for v in row)
        for value in values:
            self.assertTrue(math.isfinite(value))


if __name__ == "__main__":
    unittest.main()
