import math
import unittest

from mappilot.geometry import Pose3
from mappilot.pose_graph import PoseGraphResult, optimize_pose_graph


def rotation_z(angle):
    return Pose3(
        (0.0, 0.0, 0.0),
        (math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0)),
    )


def identity_information():
    return [[1.0 if i == j else 0.0 for j in range(6)] for i in range(6)]


def relative(pose_from, pose_to):
    return pose_from.inverse().compose(pose_to)


def chain_truth(count, step=0.3):
    return [
        Pose3((float(k), 0.2 * k, 0.1 * k), rotation_z(step * k).quaternion)
        for k in range(count)
    ]


def chain_constraints(truth, information):
    constraints = []
    for k in range(len(truth) - 1):
        constraints.append(
            (k, k + 1, relative(truth[k], truth[k + 1]), [row[:] for row in information])
        )
    return constraints


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


class OptimizePoseGraphTest(unittest.TestCase):
    def setUp(self):
        self.information = identity_information()
        self.truth = chain_truth(6)
        self.constraints = chain_constraints(self.truth, self.information)

    def initial_guesses(self, seed=0):
        # Deterministic, non-random perturbations.
        nodes = []
        for k, pose in enumerate(self.truth):
            if k == 0:
                nodes.append((k, pose))
                continue
            angle = 0.04 * k + 0.01 * seed
            delta = (angle, 0.0, 0.02 * k, 0.05, -0.03 * k, 0.01 * k)
            nodes.append((k, Pose3.exp(delta).compose(pose)))
        return nodes

    def test_recovers_consistent_trajectory(self):
        result = optimize_pose_graph(
            self.initial_guesses(), self.constraints, [0], tolerance=1e-10
        )
        self.assertTrue(result.converged)
        self.assertLess(result.final_error, 1e-14)
        self.assertLess(result.final_error, result.initial_error)
        self.assertEqual(len(result.nodes), len(self.truth))
        for index, ((node_id, pose), expected) in enumerate(
            zip(result.nodes, self.truth)
        ):
            self.assertEqual(node_id, index)
            pose_close(self, pose, expected, tol=1e-7)

    def test_loop_closure(self):
        constraints = self.constraints + [
            (0, 5, relative(self.truth[0], self.truth[5]),
             [row[:] for row in self.information])
        ]
        result = optimize_pose_graph(
            self.initial_guesses(), constraints, [0], tolerance=1e-10
        )
        self.assertTrue(result.converged)
        self.assertLess(result.final_error, 1e-14)
        for (_, pose), expected in zip(result.nodes, self.truth):
            pose_close(self, pose, expected, tol=1e-7)

    def test_nodes_returned_in_input_order(self):
        nodes = self.initial_guesses()
        result = optimize_pose_graph(nodes, self.constraints, [0])
        self.assertEqual(
            [node_id for node_id, _ in result.nodes], [node_id for node_id, _ in nodes]
        )

    def test_non_contiguous_ids_preserved(self):
        truth = chain_truth(4)
        constraints = [
            (10, 20, relative(truth[0], truth[1]), self.information),
            (20, 30, relative(truth[1], truth[2]), self.information),
            (30, 40, relative(truth[2], truth[3]), self.information),
        ]
        nodes = [(10, truth[0]), (20, Pose3.exp((0.1, 0, 0, 0.1, 0, 0)).compose(truth[1])),
                 (30, truth[2]), (40, truth[3])]
        result = optimize_pose_graph(nodes, constraints, [10], tolerance=1e-10)
        self.assertEqual([node_id for node_id, _ in result.nodes], [10, 20, 30, 40])
        self.assertTrue(result.converged)
        pose_close(self, result.nodes[1][1], truth[1], tol=1e-8)

    def test_already_consistent_graph_zero_iterations(self):
        nodes = [(k, pose) for k, pose in enumerate(self.truth)]
        result = optimize_pose_graph(nodes, self.constraints, [0])
        self.assertTrue(result.converged)
        self.assertEqual(result.iterations, 0)
        self.assertEqual(result.initial_error, 0.0)
        self.assertEqual(result.final_error, 0.0)
        for (_, actual), expected in zip(result.nodes, self.truth):
            self.assertEqual(actual, expected)

    def test_initial_error_within_tolerance_short_circuits(self):
        # A graph with a tiny but non-zero residual error and a generous
        # tolerance must report convergence without any update.
        information = [[1.0 if i == j else 0.0 for j in range(6)] for i in range(6)]
        nodes = [(0, Pose3.identity()), (1, Pose3((1.0, 0.0, 0.0)))]
        measurement = Pose3((1.0 + 1e-5, 0.0, 0.0))
        result = optimize_pose_graph(
            nodes,
            [(0, 1, measurement, information)],
            [0],
            tolerance=1e-3,
        )
        self.assertTrue(result.converged)
        self.assertEqual(result.iterations, 0)
        self.assertEqual(result.nodes[1][1], nodes[1][1])
        self.assertAlmostEqual(result.initial_error, result.final_error, places=18)
        self.assertGreater(result.initial_error, 0.0)

    def test_max_iterations_exhaustion_not_converged(self):
        result = optimize_pose_graph(
            self.initial_guesses(), self.constraints, [0],
            max_iterations=1, tolerance=1e-12,
        )
        self.assertFalse(result.converged)
        self.assertEqual(result.iterations, 1)
        # Errors are still recomputed from the returned poses.
        self.assertAlmostEqual(
            result.final_error,
            self._recomputed_error(result.nodes, self.constraints),
            places=12,
        )

    def test_convergence_on_error_plateau(self):
        # All nodes fixed: no update is possible, so after one update the
        # error change is zero and the graph converges even though a residual
        # remains.
        measurement = Pose3((1.5, 0.0, 0.0))
        nodes = [(0, Pose3.identity()), (1, Pose3((1.0, 0.0, 0.0)))]
        result = optimize_pose_graph(
            nodes, [(0, 1, measurement, self.information)], [0, 1],
            tolerance=1e-8,
        )
        self.assertTrue(result.converged)
        self.assertEqual(result.iterations, 1)
        self.assertEqual(result.nodes[0][1], nodes[0][1])
        self.assertEqual(result.nodes[1][1], nodes[1][1])

    def test_fixed_nodes_unchanged_value_by_value(self):
        # Fix nodes 0 and 3 at their *true* poses; perturb only free nodes, so
        # the remaining free nodes still converge to the truth while both
        # anchors stay bit-for-bit identical to the input objects.
        nodes = []
        for k, pose in enumerate(self.truth):
            if k in (0, 3):
                nodes.append((k, pose))
            else:
                delta = (0.04 * k, 0.0, 0.02 * k, 0.05, -0.03 * k, 0.01 * k)
                nodes.append((k, Pose3.exp(delta).compose(pose)))
        fixed_poses = {0: nodes[0][1], 3: nodes[3][1]}
        result = optimize_pose_graph(
            nodes, self.constraints, [3, 0], tolerance=1e-10
        )
        self.assertTrue(result.converged)
        for node_id, pose in result.nodes:
            if node_id in fixed_poses:
                self.assertIs(pose, fixed_poses[node_id])
        pose_close(self, result.nodes[5][1], self.truth[5], tol=1e-7)

    def test_multiple_fixed_components(self):
        truth_a = chain_truth(3)
        truth_b = chain_truth(2)
        constraints = chain_constraints(truth_a, self.information)
        constraints.append(
            (10, 11, relative(truth_b[0], truth_b[1]),
             [row[:] for row in self.information])
        )
        nodes = [(k, pose) for k, pose in enumerate(truth_a)]
        nodes.append((10, truth_b[0]))
        nodes.append(
            (11, Pose3.exp((0.0, 0.0, 0.0, 0.2, 0.0, 0.0)).compose(truth_b[1]))
        )
        result = optimize_pose_graph(nodes, constraints, [0, 10], tolerance=1e-10)
        self.assertTrue(result.converged)
        self.assertLess(result.final_error, 1e-14)
        pose_close(self, result.nodes[-1][1], truth_b[1], tol=1e-8)

    def test_isolated_fixed_node_allowed(self):
        nodes = [(0, Pose3.identity()), (1, Pose3((1.0, 0.0, 0.0))),
                 (9, Pose3((5.0, 5.0, 5.0)))]
        result = optimize_pose_graph(
            nodes, self.constraints[:1], [0, 9],
        )
        self.assertEqual(result.nodes[2][1], nodes[2][1])

    def test_errors_recomputed_from_poses(self):
        initial_nodes = self.initial_guesses()
        result = optimize_pose_graph(
            initial_nodes, self.constraints, [0], tolerance=1e-10
        )
        self.assertAlmostEqual(
            result.initial_error,
            self._recomputed_error(initial_nodes, self.constraints),
            places=12,
        )
        self.assertAlmostEqual(
            result.final_error,
            self._recomputed_error(result.nodes, self.constraints),
            places=12,
        )

    def test_information_weights_constraints(self):
        # Two edges between the same fixed/free pair with conflicting
        # measurements: the strongly weighted edge must dominate the
        # solution, so the weakly weighted residual stays large.
        strong = [[100.0 if i == j else 0.0 for j in range(6)] for i in range(6)]
        weak = [[1.0 if i == j else 0.0 for j in range(6)] for i in range(6)]
        close_measurement = Pose3((1.0, 0.0, 0.0))
        far_measurement = Pose3((2.0, 0.0, 0.0))
        nodes = [(0, Pose3.identity()), (1, Pose3((1.0, 0.0, 0.0)))]
        result = optimize_pose_graph(
            nodes,
            [(0, 1, close_measurement, strong),
             (0, 1, far_measurement, weak)],
            [0],
            tolerance=1e-12,
        )
        self.assertTrue(result.converged)
        x = result.nodes[1][1].translation[0]
        # Weighted least squares optimum: x = (100*1 + 1*2)/101 ~= 1.0099
        self.assertAlmostEqual(x, 102.0 / 101.0, places=6)
        self.assertLess(abs(x - 1.0), 0.02)
        self.assertGreater(abs(x - 2.0), 0.9)

    def test_self_loop_constraint(self):
        # A constraint from a node to itself with identity measurement is a
        # valid zero residual.
        nodes = [(0, Pose3.identity())]
        result = optimize_pose_graph(
            nodes, [(0, 0, Pose3.identity(), self.information)], [0]
        )
        self.assertTrue(result.converged)
        self.assertEqual(result.final_error, 0.0)

    def test_result_is_immutable(self):
        result = optimize_pose_graph(
            [(0, Pose3.identity())],
            [(0, 0, Pose3.identity(), self.information)],
            [0],
        )
        self.assertIsInstance(result, PoseGraphResult)
        with self.assertRaises(AttributeError):
            result.converged = False  # type: ignore[misc]

    def test_generators_accepted(self):
        result = optimize_pose_graph(
            (pair for pair in [(0, Pose3.identity()), (1, Pose3((1.0, 0.0, 0.0)))]),
            (item for item in [(0, 1, Pose3((1.0, 0.0, 0.0)), self.information)]),
            (node_id for node_id in [0]),
        )
        self.assertTrue(result.converged)

    # -- helpers -------------------------------------------------------------

    def _constraint_error(self, pose_from, pose_to, measurement, information):
        residual = measurement.inverse().compose(
            relative(pose_from, pose_to)
        ).log()
        return sum(
            residual[i]
            * sum(information[i][k] * residual[k] for k in range(6))
            for i in range(6)
        )

    def _recomputed_error(self, nodes, constraints):
        poses = dict(nodes)
        return sum(
            self._constraint_error(poses[a], poses[b], z, info)
            for a, b, z, info in constraints
        )


class OptimizePoseGraphValidationTest(unittest.TestCase):
    def setUp(self):
        self.information = identity_information()
        self.nodes = [(0, Pose3.identity()), (1, Pose3((1.0, 0.0, 0.0)))]
        self.constraints = [
            (0, 1, Pose3((1.0, 0.0, 0.0)), self.information)
        ]

    def assert_raises(self, exception, **kwargs):
        with self.assertRaises(exception):
            optimize_pose_graph(**kwargs)

    def test_non_iterable_inputs_type_error(self):
        self.assert_raises(TypeError, nodes=123,
                           constraints=self.constraints, fixed_node_ids=[0])
        self.assert_raises(TypeError, nodes=self.nodes,
                           constraints=123, fixed_node_ids=[0])
        self.assert_raises(TypeError, nodes=self.nodes,
                           constraints=self.constraints, fixed_node_ids=123)

    def test_bad_node_id_type_error(self):
        self.assert_raises(
            TypeError,
            nodes=[(True, Pose3.identity()), (1, Pose3.identity())],
            constraints=self.constraints, fixed_node_ids=[True, 1],
        )
        self.assert_raises(
            TypeError,
            nodes=[("0", Pose3.identity()), (1, Pose3.identity())],
            constraints=self.constraints, fixed_node_ids=[0],
        )
        self.assert_raises(
            TypeError,
            nodes=self.nodes, constraints=self.constraints,
            fixed_node_ids=[0.0],
        )

    def test_bad_pose_type_error(self):
        self.assert_raises(
            TypeError,
            nodes=[(0, "pose"), (1, Pose3.identity())],
            constraints=self.constraints, fixed_node_ids=[0],
        )

    def test_malformed_node_pair_type_error(self):
        self.assert_raises(TypeError, nodes=[(0,)],
                           constraints=self.constraints, fixed_node_ids=[0])
        self.assert_raises(TypeError, nodes=[(0, Pose3.identity(), 1)],
                           constraints=self.constraints, fixed_node_ids=[0])
        self.assert_raises(TypeError, nodes=[1],
                           constraints=self.constraints, fixed_node_ids=[0])

    def test_malformed_constraint_type_error(self):
        self.assert_raises(TypeError, nodes=self.nodes,
                           constraints=[(0, 1, Pose3.identity())],
                           fixed_node_ids=[0])
        self.assert_raises(TypeError, nodes=self.nodes,
                           constraints=[(0, 1, Pose3.identity(),
                                         self.information, 9)],
                           fixed_node_ids=[0])
        self.assert_raises(TypeError, nodes=self.nodes,
                           constraints=[(0.5, 1, Pose3.identity(),
                                         self.information)],
                           fixed_node_ids=[0])

    def test_bad_measurement_type_error(self):
        self.assert_raises(
            TypeError, nodes=self.nodes,
            constraints=[(0, 1, "measurement", self.information)],
            fixed_node_ids=[0],
        )

    def test_information_shape_type_error(self):
        self.assert_raises(
            TypeError, nodes=self.nodes,
            constraints=[(0, 1, Pose3.identity(), self.information[0])],
            fixed_node_ids=[0],
        )
        short_rows = [row[:5] for row in self.information]
        self.assert_raises(
            TypeError, nodes=self.nodes,
            constraints=[(0, 1, Pose3.identity(), short_rows)],
            fixed_node_ids=[0],
        )
        five_by_six = [list(row) for row in self.information[:5]]
        self.assert_raises(
            TypeError, nodes=self.nodes,
            constraints=[(0, 1, Pose3.identity(), five_by_six)],
            fixed_node_ids=[0],
        )

    def test_information_entry_type_error(self):
        information = [row[:] for row in self.information]
        information[0][0] = True
        self.assert_raises(
            TypeError, nodes=self.nodes,
            constraints=[(0, 1, Pose3.identity(), information)],
            fixed_node_ids=[0],
        )
        information2 = [row[:] for row in self.information]
        information2[2][3] = 1 + 2j
        self.assert_raises(
            TypeError, nodes=self.nodes,
            constraints=[(0, 1, Pose3.identity(), information2)],
            fixed_node_ids=[0],
        )

    def test_parameter_type_errors(self):
        kwargs = dict(nodes=self.nodes, constraints=self.constraints,
                      fixed_node_ids=[0])
        self.assert_raises(TypeError, max_iterations=1.5, **kwargs)
        self.assert_raises(TypeError, max_iterations=True, **kwargs)
        self.assert_raises(TypeError, max_iterations="50", **kwargs)
        self.assert_raises(TypeError, tolerance="1e-6", **kwargs)
        self.assert_raises(TypeError, tolerance=True, **kwargs)

    def test_empty_nodes_value_error(self):
        self.assert_raises(ValueError, nodes=[],
                           constraints=self.constraints, fixed_node_ids=[0])

    def test_duplicate_node_id_value_error(self):
        self.assert_raises(
            ValueError,
            nodes=[(0, Pose3.identity()), (0, Pose3.identity())],
            constraints=self.constraints, fixed_node_ids=[0],
        )

    def test_unknown_constraint_node_value_error(self):
        self.assert_raises(
            ValueError, nodes=self.nodes,
            constraints=[(0, 2, Pose3.identity(), self.information)],
            fixed_node_ids=[0],
        )
        self.assert_raises(
            ValueError, nodes=self.nodes,
            constraints=[(2, 1, Pose3.identity(), self.information)],
            fixed_node_ids=[0],
        )

    def test_unknown_fixed_node_value_error(self):
        self.assert_raises(ValueError, nodes=self.nodes,
                           constraints=self.constraints, fixed_node_ids=[2])

    def test_empty_fixed_set_value_error(self):
        self.assert_raises(ValueError, nodes=self.nodes,
                           constraints=self.constraints, fixed_node_ids=[])

    def test_unfixed_component_value_error(self):
        nodes = self.nodes + [(2, Pose3.identity()), (3, Pose3.identity())]
        constraints = self.constraints + [
            (2, 3, Pose3.identity(), self.information)
        ]
        # Component {0,1} is fixed; component {2,3} is not.
        self.assert_raises(ValueError, nodes=nodes,
                           constraints=constraints, fixed_node_ids=[0])
        # Isolated node without constraints is its own component.
        self.assert_raises(
            ValueError, nodes=self.nodes + [(7, Pose3.identity())],
            constraints=self.constraints, fixed_node_ids=[0],
        )

    def test_non_finite_matrix_entry_value_error(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            information = [row[:] for row in self.information]
            information[1][1] = bad
            self.assert_raises(
                ValueError, nodes=self.nodes,
                constraints=[(0, 1, Pose3.identity(), information)],
                fixed_node_ids=[0],
            )

    def test_asymmetric_matrix_value_error(self):
        information = [row[:] for row in self.information]
        information[0][1] = 2e-12
        self.assert_raises(
            ValueError, nodes=self.nodes,
            constraints=[(0, 1, Pose3.identity(), information)],
            fixed_node_ids=[0],
        )

    def test_asymmetry_within_tolerance_accepted(self):
        information = [row[:] for row in self.information]
        information[0][1] = 5e-13
        result = optimize_pose_graph(
            self.nodes,
            [(0, 1, Pose3((1.0, 0.0, 0.0)), information)],
            [0],
        )
        self.assertTrue(result.converged)

    def test_non_positive_definite_matrix_value_error(self):
        information = [row[:] for row in self.information]
        information[0][0] = 0.0
        self.assert_raises(
            ValueError, nodes=self.nodes,
            constraints=[(0, 1, Pose3.identity(), information)],
            fixed_node_ids=[0],
        )
        negative = [row[:] for row in self.information]
        negative[5][5] = -1.0
        self.assert_raises(
            ValueError, nodes=self.nodes,
            constraints=[(0, 1, Pose3.identity(), negative)],
            fixed_node_ids=[0],
        )
        # Indefinite matrix with all-positive diagonal but a negative pivot.
        indefinite = [
            [1.0 if i == j else 0.0 for j in range(6)] for i in range(6)
        ]
        indefinite[0][1] = indefinite[1][0] = 2.0
        self.assert_raises(
            ValueError, nodes=self.nodes,
            constraints=[(0, 1, Pose3.identity(), indefinite)],
            fixed_node_ids=[0],
        )

    def test_bad_parameter_values(self):
        kwargs = dict(nodes=self.nodes, constraints=self.constraints,
                      fixed_node_ids=[0])
        self.assert_raises(ValueError, max_iterations=0, **kwargs)
        self.assert_raises(ValueError, max_iterations=-5, **kwargs)
        self.assert_raises(ValueError, tolerance=0.0, **kwargs)
        self.assert_raises(ValueError, tolerance=-1e-6, **kwargs)
        self.assert_raises(ValueError, tolerance=float("inf"), **kwargs)
        self.assert_raises(ValueError, tolerance=float("nan"), **kwargs)

    def test_defaults_are_50_and_1e_6(self):
        import inspect

        signature = inspect.signature(optimize_pose_graph)
        self.assertEqual(signature.parameters["max_iterations"].default, 50)
        self.assertEqual(signature.parameters["tolerance"].default, 1e-6)


if __name__ == "__main__":
    unittest.main()
