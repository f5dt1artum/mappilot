import math
import unittest

from mappilot.geometry import Pose3
from mappilot.pose_graph import PoseGraphResult, optimize_pose_graph

INFO = tuple(
    tuple(1.0 if i == j else 0.0 for j in range(6)) for i in range(6)
)


def noisy_pose(pose: Pose3, seed: int) -> Pose3:
    """Deterministically perturb a pose for use as an initial guess."""
    xi = tuple(0.01 * math.sin(3.1 * seed + k) for k in range(6))
    return Pose3.exp(xi).compose(pose)


def chain_problem(count: int = 6):
    """Ground-truth chain (with yaw rotation) plus one loop closure."""
    truth = []
    for i in range(count):
        angle = 0.15 * i
        quaternion = (math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0))
        truth.append(
            Pose3((0.5 * i, 0.1 * math.sin(i), 0.02 * i), quaternion)
        )
    constraints = []
    for i in range(count - 1):
        measurement = truth[i].inverse().compose(truth[i + 1])
        constraints.append((i, i + 1, measurement, INFO))
    measurement = truth[0].inverse().compose(truth[count - 1])
    constraints.append((0, count - 1, measurement, INFO))
    return truth, constraints


class OptimizePoseGraphTest(unittest.TestCase):
    def test_chain_converges_to_ground_truth(self) -> None:
        truth, constraints = chain_problem()
        nodes = [(0, truth[0])] + [
            (i, noisy_pose(truth[i], i)) for i in range(1, len(truth))
        ]
        result = optimize_pose_graph(nodes, constraints, [0], tolerance=1e-12)
        self.assertTrue(result.converged)
        self.assertGreater(result.iterations, 0)
        self.assertLess(result.final_error, 1e-16)
        self.assertLess(result.final_error, result.initial_error)
        for (node_id, pose), expected in zip(result.poses, truth):
            self.assertEqual(node_id, truth.index(expected))
            for actual, wanted in zip(pose.translation, expected.translation):
                self.assertAlmostEqual(actual, wanted, places=8)
            for actual, wanted in zip(pose.quaternion, expected.quaternion):
                self.assertAlmostEqual(actual, wanted, places=8)

    def test_result_is_immutable_and_ordered(self) -> None:
        truth, constraints = chain_problem(4)
        nodes = [(i, noisy_pose(truth[i], i)) for i in range(len(truth))]
        result = optimize_pose_graph(nodes, constraints, [0])
        self.assertIsInstance(result, PoseGraphResult)
        self.assertIsInstance(result.poses, tuple)
        self.assertEqual([node_id for node_id, _ in result.poses], [0, 1, 2, 3])
        with self.assertRaises(AttributeError):
            result.converged = False  # type: ignore[misc]

    def test_fixed_nodes_are_returned_value_identical(self) -> None:
        truth, constraints = chain_problem(5)
        fixed_pose = truth[2]
        nodes = [
            (i, truth[i] if i == 2 else noisy_pose(truth[i], i))
            for i in range(len(truth))
        ]
        result = optimize_pose_graph(nodes, constraints, [2])
        returned = dict(result.poses)[2]
        self.assertIs(returned, fixed_pose)
        self.assertEqual(returned.translation, fixed_pose.translation)
        self.assertEqual(returned.quaternion, fixed_pose.quaternion)

    def test_initial_error_within_tolerance_short_circuits(self) -> None:
        truth, constraints = chain_problem(4)
        nodes = [(i, truth[i]) for i in range(len(truth))]
        result = optimize_pose_graph(nodes, constraints, [0])
        self.assertTrue(result.converged)
        self.assertEqual(result.iterations, 0)
        self.assertEqual(result.initial_error, 0.0)
        self.assertEqual(result.final_error, result.initial_error)
        self.assertEqual([node_id for node_id, _ in result.poses], [0, 1, 2, 3])

    def test_max_iterations_exhausted_reports_not_converged(self) -> None:
        truth, constraints = chain_problem()
        nodes = [(0, truth[0])] + [
            (i, noisy_pose(truth[i], i)) for i in range(1, len(truth))
        ]
        result = optimize_pose_graph(
            nodes, constraints, [0], max_iterations=1, tolerance=1e-15
        )
        self.assertFalse(result.converged)
        self.assertEqual(result.iterations, 1)
        self.assertLess(result.final_error, result.initial_error)

    def test_information_weights_the_solution(self) -> None:
        # Two conflicting translation-only constraints between the same nodes;
        # the optimum is the information-weighted average of the measurements.
        identity = Pose3.identity()
        nodes = [(0, identity), (1, identity)]
        info_one = INFO
        info_three = tuple(
            tuple(3.0 if i == j else 0.0 for j in range(6)) for i in range(6)
        )
        constraints = [
            (0, 1, Pose3((1.0, 0.0, 0.0)), info_one),
            (0, 1, Pose3((3.0, 0.0, 0.0)), info_three),
        ]
        result = optimize_pose_graph(nodes, constraints, [0], tolerance=1e-14)
        self.assertTrue(result.converged)
        x = dict(result.poses)[1].translation[0]
        self.assertAlmostEqual(x, 2.5, places=10)

    def test_multiple_components_each_need_a_fixed_node(self) -> None:
        pose = Pose3.identity()
        nodes = [(0, pose), (1, pose), (2, pose), (3, pose)]
        constraints = [
            (0, 1, pose, INFO),
            (2, 3, pose, INFO),
        ]
        with self.assertRaises(ValueError):
            optimize_pose_graph(nodes, constraints, [0])
        result = optimize_pose_graph(nodes, constraints, [0, 2])
        self.assertTrue(result.converged)

    def test_isolated_node_must_be_fixed(self) -> None:
        pose = Pose3.identity()
        nodes = [(0, pose), (1, pose)]
        with self.assertRaises(ValueError):
            optimize_pose_graph(nodes, [], [0])
        result = optimize_pose_graph(nodes, [], [0, 1])
        self.assertTrue(result.converged)
        self.assertEqual(result.iterations, 0)


class ValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.pose = Pose3.identity()
        self.nodes = [(0, self.pose), (1, Pose3((1.0, 0.0, 0.0)))]
        self.constraints = [(0, 1, Pose3((1.0, 0.0, 0.0)), INFO)]

    def test_type_errors(self) -> None:
        with self.assertRaises(TypeError):
            optimize_pose_graph(42, self.constraints, [0])
        with self.assertRaises(TypeError):
            optimize_pose_graph(self.nodes, 42, [0])
        with self.assertRaises(TypeError):
            optimize_pose_graph(self.nodes, self.constraints, 42)
        with self.assertRaises(TypeError):
            optimize_pose_graph([("a", self.pose)], [], ["a"])
        with self.assertRaises(TypeError):
            optimize_pose_graph([(True, self.pose)], [], [True])
        with self.assertRaises(TypeError):
            optimize_pose_graph([(0, "not a pose")], [], [0])
        with self.assertRaises(TypeError):
            optimize_pose_graph([(0, self.pose, 1)], [], [0])
        with self.assertRaises(TypeError):
            optimize_pose_graph(self.nodes, [(0, 1, self.pose)], [0])
        with self.assertRaises(TypeError):
            optimize_pose_graph(
                self.nodes, [(0, 1, self.pose, [[1.0] * 6] * 5)], [0]
            )
        with self.assertRaises(TypeError):
            optimize_pose_graph(
                self.nodes, [(0, 1, self.pose, [[1.0] * 5] * 6)], [0]
            )
        with self.assertRaises(TypeError):
            optimize_pose_graph(
                self.nodes, [(0, 1, self.pose, [[True] * 6] * 6)], [0]
            )
        with self.assertRaises(TypeError):
            optimize_pose_graph(
                self.nodes, [(0, 1, self.pose, [["x"] * 6] * 6)], [0]
            )
        with self.assertRaises(TypeError):
            optimize_pose_graph(self.nodes, [(0, 1, "not a pose", INFO)], [0])
        with self.assertRaises(TypeError):
            optimize_pose_graph(
                self.nodes, self.constraints, [0], max_iterations=1.5
            )
        with self.assertRaises(TypeError):
            optimize_pose_graph(
                self.nodes, self.constraints, [0], max_iterations=True
            )
        with self.assertRaises(TypeError):
            optimize_pose_graph(
                self.nodes, self.constraints, [0], tolerance="1e-6"
            )

    def test_value_errors(self) -> None:
        with self.assertRaises(ValueError):
            optimize_pose_graph([], self.constraints, [0])
        with self.assertRaises(ValueError):
            optimize_pose_graph([(0, self.pose), (0, self.pose)], [], [0])
        with self.assertRaises(ValueError):
            optimize_pose_graph(self.nodes, [(0, 99, self.pose, INFO)], [0])
        with self.assertRaises(ValueError):
            optimize_pose_graph(self.nodes, self.constraints, [0, 99])
        with self.assertRaises(ValueError):
            optimize_pose_graph(self.nodes, self.constraints, [])
        asymmetric = [list(row) for row in INFO]
        asymmetric[0][1] = 1e-6
        with self.assertRaises(ValueError):
            optimize_pose_graph(self.nodes, [(0, 1, self.pose, asymmetric)], [0])
        not_pd = [[0.0] * 6 for _ in range(6)]
        with self.assertRaises(ValueError):
            optimize_pose_graph(self.nodes, [(0, 1, self.pose, not_pd)], [0])
        negative = [list(row) for row in INFO]
        negative[3][3] = -1.0
        with self.assertRaises(ValueError):
            optimize_pose_graph(self.nodes, [(0, 1, self.pose, negative)], [0])
        nan_matrix = [list(row) for row in INFO]
        nan_matrix[2][2] = float("nan")
        with self.assertRaises(ValueError):
            optimize_pose_graph(self.nodes, [(0, 1, self.pose, nan_matrix)], [0])
        inf_matrix = [list(row) for row in INFO]
        inf_matrix[0][0] = float("inf")
        with self.assertRaises(ValueError):
            optimize_pose_graph(self.nodes, [(0, 1, self.pose, inf_matrix)], [0])
        with self.assertRaises(ValueError):
            optimize_pose_graph(
                self.nodes, self.constraints, [0], max_iterations=0
            )
        with self.assertRaises(ValueError):
            optimize_pose_graph(
                self.nodes, self.constraints, [0], max_iterations=-3
            )
        with self.assertRaises(ValueError):
            optimize_pose_graph(self.nodes, self.constraints, [0], tolerance=0.0)
        with self.assertRaises(ValueError):
            optimize_pose_graph(self.nodes, self.constraints, [0], tolerance=-1.0)
        with self.assertRaises(ValueError):
            optimize_pose_graph(
                self.nodes, self.constraints, [0], tolerance=float("nan")
            )
        with self.assertRaises(ValueError):
            optimize_pose_graph(
                self.nodes, self.constraints, [0], tolerance=float("inf")
            )

    def test_symmetry_tolerance_boundary(self) -> None:
        # A deviation well below 1e-12 is accepted.
        slightly_off = [list(row) for row in INFO]
        slightly_off[0][1] = 1e-14
        result = optimize_pose_graph(
            self.nodes, [(0, 1, self.pose, slightly_off)], [0]
        )
        self.assertTrue(result.converged)

    def test_defaults_match_icp_defaults(self) -> None:
        import inspect

        signature = inspect.signature(optimize_pose_graph)
        self.assertEqual(signature.parameters["max_iterations"].default, 50)
        self.assertEqual(signature.parameters["tolerance"].default, 1e-6)


if __name__ == "__main__":
    unittest.main()
