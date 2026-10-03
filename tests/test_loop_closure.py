import math
import unittest

from mappilot.geometry import Pose3
from mappilot.loop_closure import LoopClosure, detect_loop_closures


# Non-degenerate world point set observed by every keyframe.
WORLD_POINTS = (
    (0.0, 0.0, 0.0),
    (1.0, 0.2, 0.1),
    (0.3, 1.1, 0.2),
    (0.1, 0.4, 1.2),
    (1.3, 1.1, 0.9),
    (0.7, 0.3, 1.5),
    (1.6, 0.6, 0.4),
    (0.2, 1.5, 1.3),
)


def make_frame(frame_id, translation, quaternion=(1.0, 0.0, 0.0, 0.0)):
    """Keyframe observing WORLD_POINTS from the given world pose."""
    pose = Pose3(translation, quaternion)
    cloud = pose.inverse().transform_points(WORLD_POINTS)
    return (frame_id, pose, cloud)


def trajectory():
    """Frames moving away and returning close to the first frame's pose."""
    return [
        make_frame(0, (0.0, 0.0, 0.0)),
        make_frame(1, (5.0, 0.0, 0.0)),
        make_frame(2, (10.0, 0.0, 0.0)),
        make_frame(3, (15.0, 0.0, 0.0)),
        make_frame(4, (10.0, 5.0, 0.0)),
        make_frame(5, (5.0, 5.0, 0.0)),
        make_frame(6, (0.2, 0.1, 0.0)),
    ]


class DetectLoopClosuresTest(unittest.TestCase):
    def test_empty_input_returns_empty_tuple(self) -> None:
        self.assertEqual(detect_loop_closures([]), ())
        self.assertEqual(detect_loop_closures(iter([])), ())

    def test_detects_return_to_start(self) -> None:
        closures = detect_loop_closures(
            trajectory(),
            min_separation=3,
            translation_threshold=2.0,
            rotation_threshold=math.pi / 2.0,
        )
        self.assertEqual(len(closures), 1)
        closure = closures[0]
        self.assertIsInstance(closure, LoopClosure)
        self.assertEqual(closure.early_id, 0)
        self.assertEqual(closure.late_id, 6)
        self.assertLess(closure.rmse, 1e-6)
        self.assertEqual(
            len(closure.correspondences), len(WORLD_POINTS)
        )
        # measurement convention: T_early^-1 * T_late
        expected = Pose3((0.0, 0.0, 0.0)).inverse().compose(Pose3((0.2, 0.1, 0.0)))
        for actual, wanted in zip(
            closure.relative_pose.translation, expected.translation
        ):
            self.assertAlmostEqual(actual, wanted, places=9)
        for actual, wanted in zip(
            closure.relative_pose.quaternion, expected.quaternion
        ):
            self.assertAlmostEqual(actual, wanted, places=9)

    def test_measurement_is_inverse_of_scan_transform(self) -> None:
        # A rotated return pose exercises the rotation part of the edge.
        half = math.pi / 4.0
        frames = [
            make_frame(0, (0.0, 0.0, 0.0)),
            make_frame(1, (8.0, 0.0, 0.0)),
            make_frame(2, (0.3, -0.2, 0.1), (math.cos(half), 0.0, 0.0, math.sin(half))),
        ]
        closures = detect_loop_closures(
            frames,
            min_separation=2,
            translation_threshold=2.0,
            rotation_threshold=math.pi,
        )
        self.assertEqual(len(closures), 1)
        expected = frames[0][1].inverse().compose(frames[2][1])
        for actual, wanted in zip(
            closures[0].relative_pose.translation, expected.translation
        ):
            self.assertAlmostEqual(actual, wanted, places=8)
        for actual, wanted in zip(
            closures[0].relative_pose.quaternion, expected.quaternion
        ):
            self.assertAlmostEqual(actual, wanted, places=8)

    def test_results_sorted_by_input_position(self) -> None:
        frames = [
            make_frame(0, (0.0, 0.0, 0.0)),
            make_frame(1, (10.0, 0.0, 0.0)),
            make_frame(2, (20.0, 0.0, 0.0)),
            make_frame(3, (30.0, 0.0, 0.0)),
            make_frame(4, (0.1, 0.0, 0.0)),
            make_frame(5, (10.1, 0.0, 0.0)),
        ]
        closures = detect_loop_closures(
            frames,
            min_separation=3,
            translation_threshold=1.0,
            rotation_threshold=math.pi,
        )
        self.assertEqual(
            [(c.early_id, c.late_id) for c in closures], [(0, 4), (1, 5)]
        )

    def test_generator_input_consumed_once(self) -> None:
        frames = trajectory()
        closures = detect_loop_closures(
            (frame for frame in frames),
            min_separation=3,
            translation_threshold=2.0,
            rotation_threshold=math.pi / 2.0,
        )
        self.assertEqual(len(closures), 1)

    def test_min_separation_filters_close_positions(self) -> None:
        closures = detect_loop_closures(
            trajectory(),
            min_separation=7,
            translation_threshold=100.0,
            rotation_threshold=math.pi,
        )
        self.assertEqual(closures, ())

    def test_translation_threshold_filters_distant_pairs(self) -> None:
        closures = detect_loop_closures(
            trajectory(),
            min_separation=3,
            translation_threshold=0.1,  # return pose is 0.2 away
            rotation_threshold=math.pi,
        )
        self.assertEqual(closures, ())

    def test_rotation_threshold_filters_turned_pairs(self) -> None:
        frames = [
            make_frame(0, (0.0, 0.0, 0.0)),
            make_frame(1, (8.0, 0.0, 0.0)),
            make_frame(2, (0.1, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),  # 180 deg
        ]
        common = dict(min_separation=2, translation_threshold=2.0)
        self.assertEqual(
            detect_loop_closures(frames, rotation_threshold=math.pi / 2.0, **common),
            (),
        )
        self.assertEqual(
            len(detect_loop_closures(frames, rotation_threshold=math.pi, **common)),
            1,
        )

    def test_max_rmse_rejects_noisy_alignment(self) -> None:
        frames = trajectory()
        late_id, late_pose, late_cloud = frames[6]
        noisy = tuple(
            (x + 0.05, y, z) if index == 0 else (x, y, z)
            for index, (x, y, z) in enumerate(late_cloud)
        )
        frames[6] = (late_id, late_pose, noisy)
        common = dict(
            min_separation=3,
            translation_threshold=2.0,
            rotation_threshold=math.pi / 2.0,
        )
        self.assertEqual(detect_loop_closures(frames, max_rmse=1e-9, **common), ())
        self.assertEqual(len(detect_loop_closures(frames, max_rmse=0.5, **common)), 1)

    def test_min_correspondences_rejects_small_overlap(self) -> None:
        closures = detect_loop_closures(
            trajectory(),
            min_separation=3,
            translation_threshold=2.0,
            rotation_threshold=math.pi / 2.0,
            min_correspondences=len(WORLD_POINTS) + 1,
        )
        self.assertEqual(closures, ())

    def test_non_converged_icp_rejects_candidate(self) -> None:
        closures = detect_loop_closures(
            trajectory(),
            min_separation=3,
            translation_threshold=2.0,
            rotation_threshold=math.pi / 2.0,
            max_iterations=1,
        )
        self.assertEqual(closures, ())

    def test_degenerate_cloud_rejects_candidate(self) -> None:
        line = [(float(i), 0.0, 0.0) for i in range(5)]
        frames = [
            (0, Pose3((0.0, 0.0, 0.0)), line),
            (1, Pose3((8.0, 0.0, 0.0)), line),
            (2, Pose3((0.1, 0.0, 0.0)), line),
        ]
        closures = detect_loop_closures(
            frames,
            min_separation=2,
            translation_threshold=2.0,
            rotation_threshold=math.pi,
        )
        self.assertEqual(closures, ())

    def test_correspondence_gate_rejects_candidate(self) -> None:
        # The recorded return pose is ~1m off the pose the cloud was actually
        # observed from, so a 0.1m gate drops every initial pair.
        frames = trajectory()
        late_id, _, late_cloud = frames[6]
        frames[6] = (late_id, Pose3((1.2, 0.1, 0.0)), late_cloud)
        closures = detect_loop_closures(
            frames,
            min_separation=3,
            translation_threshold=2.0,
            rotation_threshold=math.pi / 2.0,
            max_correspondence_distance=0.1,
        )
        self.assertEqual(closures, ())

    def test_result_is_immutable(self) -> None:
        closures = detect_loop_closures(
            trajectory(),
            min_separation=3,
            translation_threshold=2.0,
            rotation_threshold=math.pi / 2.0,
        )
        self.assertIsInstance(closures, tuple)
        with self.assertRaises(AttributeError):
            closures[0].rmse = 1.0  # type: ignore[misc]

    def test_input_objects_not_modified(self) -> None:
        frames = [
            [frame_id, pose, [list(point) for point in cloud]]
            for frame_id, pose, cloud in trajectory()
        ]
        snapshot = [
            (frame_id, pose, tuple(tuple(point) for point in cloud))
            for frame_id, pose, cloud in frames
        ]
        detect_loop_closures(
            frames,
            min_separation=3,
            translation_threshold=2.0,
            rotation_threshold=math.pi / 2.0,
        )
        for (frame_id, pose, cloud), (want_id, want_pose, want_cloud) in zip(
            frames, snapshot
        ):
            self.assertEqual(frame_id, want_id)
            self.assertEqual(pose, want_pose)
            self.assertEqual(
                tuple(tuple(point) for point in cloud), want_cloud
            )


class LoopClosureValidationTest(unittest.TestCase):
    def test_type_errors(self) -> None:
        bad_keyframes = [
            42,
            "frames",
            [(0, Pose3(), WORLD_POINTS, "extra")],
            [("0", Pose3(), WORLD_POINTS)],
            [(True, Pose3(), WORLD_POINTS)],
            [(0, "not a pose", WORLD_POINTS)],
            [(0, Pose3(), "cloud")],
            [(0, Pose3(), [(0.0, 0.0)] * 3)],
            [(0, Pose3(), [(0.0, 0.0, True)] * 3)],
            [(0, Pose3(), [(0.0, 0.0, "z")] * 3)],
        ]
        for keyframes in bad_keyframes:
            with self.subTest(keyframes=keyframes):
                with self.assertRaises(TypeError):
                    detect_loop_closures(keyframes)

    def test_value_errors(self) -> None:
        bad_keyframes = [
            [(0, Pose3(), WORLD_POINTS), (0, Pose3(), WORLD_POINTS)],  # dup id
            [(0, Pose3(), [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)])],  # < 3 points
            [(0, Pose3(), [(0.0, 0.0, 0.0)] * 2 + [(math.nan, 0.0, 0.0)])],
            [(0, Pose3(), [(0.0, 0.0, 0.0)] * 2 + [(0.0, math.inf, 0.0)])],
        ]
        for keyframes in bad_keyframes:
            with self.subTest(keyframes=keyframes):
                with self.assertRaises(ValueError):
                    detect_loop_closures(keyframes)

    def test_parameter_type_errors(self) -> None:
        frames = trajectory()
        bad_parameters = [
            {"min_separation": True},
            {"min_separation": 1.5},
            {"translation_threshold": "1.0"},
            {"translation_threshold": True},
            {"rotation_threshold": None},
            {"min_correspondences": 3.0},
            {"max_rmse": "0.5"},
            {"max_iterations": 10.0},
            {"tolerance": "1e-6"},
            {"max_correspondence_distance": "1.0"},
        ]
        for parameters in bad_parameters:
            with self.subTest(parameters=parameters):
                with self.assertRaises(TypeError):
                    detect_loop_closures(frames, **parameters)

    def test_parameter_value_errors(self) -> None:
        frames = trajectory()
        bad_parameters = [
            {"min_separation": 0},
            {"min_separation": -2},
            {"translation_threshold": -0.1},
            {"translation_threshold": math.inf},
            {"translation_threshold": math.nan},
            {"rotation_threshold": -0.1},
            {"rotation_threshold": math.pi + 1e-9},
            {"rotation_threshold": math.inf},
            {"min_correspondences": 2},
            {"max_rmse": -0.1},
            {"max_rmse": math.inf},
            {"max_iterations": 0},
            {"tolerance": 0.0},
            {"max_correspondence_distance": 0.0},
        ]
        for parameters in bad_parameters:
            with self.subTest(parameters=parameters):
                with self.assertRaises(ValueError):
                    detect_loop_closures(frames, **parameters)

    def test_threshold_boundary_values_accepted(self) -> None:
        frames = trajectory()
        closures = detect_loop_closures(
            frames,
            min_separation=3,
            translation_threshold=0.0,  # allowed: non-negative
            rotation_threshold=0.0,  # allowed: zero
            max_rmse=0.0,  # allowed: non-negative
        )
        self.assertEqual(closures, ())


if __name__ == "__main__":
    unittest.main()
