from types import SimpleNamespace

from real_robot.contracts import (
    CameraFrame,
    CameraModel,
    DetectionFrame,
    NavigationStatus,
    NavigationStatusCode,
    ObjectNodeSnapshot,
    Pose3D,
    RealObservation,
    SemanticMapSnapshot,
    ViewpointGoal,
)
from real_robot.observation_cache import ObjectCropEvidenceProvider, RosObservationCache
from real_robot.sysnav_runtime import ViewpointEvidenceLoop


def _header(sec=1, nanosec=250_000_000, frame_id="camera"):
    return SimpleNamespace(stamp=SimpleNamespace(sec=sec, nanosec=nanosec), frame_id=frame_id)


def _image_msg(sec=1, width=640, height=480, encoding="rgb8", data=None):
    if data is None:
        data = bytes(width * height * 3)
    return SimpleNamespace(
        header=_header(sec=sec),
        height=height,
        width=width,
        encoding=encoding,
        step=width * 3,
        data=data,
    )


def _pointcloud_msg(sec=1):
    return SimpleNamespace(header=_header(sec=sec, frame_id="map"))


def _detection_msg():
    return SimpleNamespace(
        header=_header(sec=3),
        track_id=[7],
        x1=[100.0],
        y1=[120.0],
        x2=[220.0],
        y2=[260.0],
        label=["book"],
        confidence=[0.8],
        image=None,
    )


def test_ros_observation_cache_builds_reference_only_observation() -> None:
    cache = RosObservationCache(rgb_topic="/camera/image", pointcloud_topic="/cloud_registered")
    cache.update_pose(Pose3D(position=(1.0, 2.0, 0.0), frame_id="map", stamp=2.0))
    cache.update_rgb_image(_image_msg(sec=2), topic="/camera/image")
    cache.update_pointcloud(_pointcloud_msg(sec=2), topic="/cloud_registered")
    frame = cache.update_detection_result(_detection_msg())

    observation = cache.latest_observation()

    assert observation is not None
    assert observation.robot_pose.position == (1.0, 2.0, 0.0)
    assert observation.primary_camera().image_ref.startswith("ros:///camera/image/image/")
    assert observation.primary_camera().rgb_shape == (480, 640, 3)
    assert observation.pointcloud_ref.startswith("ros:///cloud_registered/pointcloud/")
    assert observation.metadata["has_detection_frame"] is True
    assert frame.track_ids == ("7",)
    assert cache.latest_detection_frame.labels == ("book",)


def test_ros_observation_cache_can_persist_raw_image_bytes(tmp_path) -> None:
    cache = RosObservationCache(image_directory=tmp_path, persist_images=True)
    cache.update_pose(Pose3D(position=(0.0, 0.0, 0.0), stamp=1.0))
    record = cache.update_rgb_image(_image_msg(sec=1, width=1, height=2, data=b"abcdef"))
    observation = cache.latest_observation()

    assert observation.primary_camera().image_ref == record.raw_path
    assert (tmp_path / "rgb_1_250000000.bin").read_bytes() == b"abcdef"
    assert (tmp_path / "rgb_1_250000000.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert (tmp_path / "rgb_1_250000000.json").exists()
    assert observation.primary_camera().metadata["storage"] == "file"


def test_object_crop_evidence_provider_uses_detection_bbox_and_quality_facts() -> None:
    observation_cache = RosObservationCache()
    observation_cache.update_pose(Pose3D(position=(0.0, 0.0, 0.0), stamp=2.0))
    observation_cache.update_rgb_image(_image_msg(sec=2))
    detection = DetectionFrame(
        timestamp=2.0,
        image_ref="ros:///camera/image/image/2.000000000",
        boxes_xyxy=((100.0, 120.0, 220.0, 260.0),),
        labels=("book",),
        confidences=(0.8,),
        track_ids=("7",),
    )
    observation_cache.update_detection_frame(detection)
    snapshot = SemanticMapSnapshot(
        timestamp=2.0,
        robot_pose=Pose3D(position=(0.0, 0.0, 0.0)),
        objects=(
            ObjectNodeSnapshot(
                uid="sysnav_object:7",
                label="book",
                track_ids=("7",),
            ),
        ),
    )
    provider = ObjectCropEvidenceProvider(
        observation_provider=observation_cache.latest_observation,
        object_provider=lambda: snapshot,
        detection_provider=lambda: observation_cache.latest_detection_frame,
        mode="bbox_crop",
    )

    evidence = provider.capture(
        ViewpointGoal(
            pose=Pose3D(position=(0.0, 0.0, 0.0)),
            target_object_uid="sysnav_object:7",
        ),
        NavigationStatus(NavigationStatusCode.REACHED),
    )

    assert evidence.image_ref.startswith("ros:///camera/image/image/")
    assert evidence.bbox_xyxy == (100.0, 120.0, 220.0, 260.0)
    assert evidence.quality["evidence_mode"] == "bbox_crop"
    assert evidence.quality["bbox_source"] == "detection_track"
    assert evidence.quality["bbox_area_px"] == 16800.0
    assert evidence.quality["bbox_area_ratio"] > 0.0
    assert evidence.quality["center_score"] is not None
    assert evidence.quality["border_margin_px"] == 100.0
    assert evidence.metadata["object_found"] is True


def test_object_crop_evidence_provider_supports_full_image_mode() -> None:
    observation_cache = RosObservationCache(camera_model=CameraModel.PINHOLE)
    observation_cache.update_pose(Pose3D(position=(0.0, 0.0, 0.0), stamp=2.0))
    observation_cache.update_rgb_image(_image_msg(sec=2))
    provider = ObjectCropEvidenceProvider(
        observation_provider=observation_cache.latest_observation,
        object_provider=lambda: (),
        mode="full_image",
    )

    evidence = provider.capture(
        ViewpointGoal(pose=Pose3D(position=(0.0, 0.0, 0.0))),
        NavigationStatus(NavigationStatusCode.REACHED),
    )

    assert evidence.bbox_xyxy is None
    assert evidence.camera_model == CameraModel.PINHOLE
    assert evidence.quality["evidence_mode"] == "full_image"


def test_post_reach_evidence_does_not_mix_new_rgb_with_stale_object_bbox() -> None:
    observation = RealObservation(
        timestamp=3.0,
        robot_pose=Pose3D(position=(1.0, 0.0, 0.0), stamp=3.0),
        camera_frames=(
            CameraFrame(
                image_ref="ros:///camera/image/image/3.000000000",
                camera_model=CameraModel.PINHOLE,
            ),
        ),
    )
    stale_detection = DetectionFrame(
        timestamp=2.0,
        image_ref="ros:///camera/image/image/2.000000000",
        boxes_xyxy=((10.0, 10.0, 80.0, 80.0),),
        labels=("book",),
        confidences=(0.9,),
        track_ids=("7",),
    )
    snapshot = SemanticMapSnapshot(
        timestamp=2.0,
        robot_pose=observation.robot_pose,
        objects=(
            ObjectNodeSnapshot(
                uid="sysnav_object:7",
                label="book",
                bbox2d_xyxy=(100.0, 100.0, 200.0, 200.0),
                track_ids=("7",),
            ),
        ),
    )
    provider = ObjectCropEvidenceProvider(
        observation_provider=lambda: observation,
        observation_after_provider=lambda _: observation,
        object_provider=lambda: snapshot,
        detection_provider=lambda: stale_detection,
        detection_after_provider=lambda _: stale_detection,
        mode="bbox_crop",
    )

    evidence = provider.capture_after(
        ViewpointGoal(pose=observation.robot_pose, target_object_uid="sysnav_object:7"),
        NavigationStatus(NavigationStatusCode.REACHED, stamp=2.5),
    )

    assert evidence is not None
    assert evidence.image_ref == observation.primary_camera().image_ref
    assert evidence.bbox_xyxy is None
    assert evidence.verifier_payload["bbox_source"] is None


class _FakeController:
    def send_goal(self, goal):
        return "goal-1"

    def poll_status(self, goal_id):
        return NavigationStatus(NavigationStatusCode.TIMEOUT, goal_id=goal_id)


class _CountingEvidenceProvider:
    def __init__(self):
        self.calls = []

    def capture(self, goal, status):
        self.calls.append((goal, status))
        raise AssertionError("capture must not run for timeout")


class _CountingVerifier:
    def __init__(self):
        self.calls = []

    def verify(self, evidence, context):
        self.calls.append((evidence, context))
        return {}


def test_viewpoint_evidence_loop_does_not_capture_or_verify_on_timeout() -> None:
    evidence_provider = _CountingEvidenceProvider()
    verifier = _CountingVerifier()
    loop = ViewpointEvidenceLoop(
        motion_controller=_FakeController(),
        evidence_provider=evidence_provider,
        final_verifier=verifier,
    )

    result = loop.run(ViewpointGoal(pose=Pose3D(position=(1.0, 0.0, 0.0))))

    assert result.status.status == NavigationStatusCode.TIMEOUT
    assert result.evidence is None
    assert evidence_provider.calls == []
    assert verifier.calls == []
