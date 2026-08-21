from __future__ import annotations

from dataclasses import dataclass

import pytest

from real_robot.contracts import (
    MotionGoalMode,
    NavigationIntent,
    ObjectNodeSnapshot,
    Pose3D,
    SemanticMapSnapshot,
    ViewpointSnapshot,
)
from real_robot.sysnav_goal_resolver import (
    PreResolvedGoalResolver,
    SnapshotViewpointProvider,
    SysNavGoalResolver,
)


def _snapshot() -> SemanticMapSnapshot:
    return SemanticMapSnapshot(
        timestamp=1.0,
        robot_pose=Pose3D(position=(0.0, 0.0, 0.0), frame_id="map"),
        objects=(
            ObjectNodeSnapshot(
                uid="book-1",
                label="book",
                position=(9.0, 9.0, 0.0),
                visible_viewpoints=("vp-1", "vp-2"),
            ),
        ),
        viewpoints=(
            ViewpointSnapshot(
                uid="vp-1",
                pose=Pose3D(position=(1.0, 0.0, 0.0), frame_id="map"),
                visible_objects=("book-1",),
            ),
            ViewpointSnapshot(
                uid="vp-2",
                pose=Pose3D(position=(2.0, 0.0, 0.0), frame_id="map"),
                visible_objects=("book-1",),
            ),
        ),
    )


@dataclass
class FakeSysNavProvider:
    viewpoint: ViewpointSnapshot | None

    def select_viewpoint(self, intent, snapshot, excluded_viewpoint_ids):
        assert intent.target_object_uid == "book-1"
        assert snapshot.object_by_uid("book-1") is not None
        if self.viewpoint is not None and self.viewpoint.uid in excluded_viewpoint_ids:
            return None
        return self.viewpoint


@dataclass
class NonFilteringProvider:
    """Test provider that deliberately violates the exclusion contract."""

    viewpoint: ViewpointSnapshot

    def select_viewpoint(self, intent, snapshot, excluded_viewpoint_ids):
        """Return the configured viewpoint without applying exclusions."""

        del intent, snapshot, excluded_viewpoint_ids
        return self.viewpoint


def test_sysnav_resolver_uses_viewpoint_pose_not_object_centroid() -> None:
    intent = NavigationIntent(mode=MotionGoalMode.GO_TO_OBJECT, target_object_uid="book-1")
    goal = SysNavGoalResolver(FakeSysNavProvider(_snapshot().viewpoints[0])).resolve(intent, _snapshot(), set())

    assert goal is not None
    assert goal.goal_pose.position == (1.0, 0.0, 0.0)
    assert goal.goal_pose.position != _snapshot().object_by_uid("book-1").position
    assert goal.metadata["viewpoint_uid"] == "vp-1"


def test_sysnav_resolver_forwards_excluded_viewpoints() -> None:
    provider = FakeSysNavProvider(_snapshot().viewpoints[0])
    intent = NavigationIntent(mode=MotionGoalMode.GO_TO_OBJECT, target_object_uid="book-1")

    assert SysNavGoalResolver(provider).resolve(intent, _snapshot(), {"vp-1"}) is None


def test_sysnav_resolver_returns_none_when_sysnav_has_no_candidate() -> None:
    intent = NavigationIntent(mode=MotionGoalMode.GO_TO_OBJECT, target_object_uid="book-1")

    assert SysNavGoalResolver(FakeSysNavProvider(None)).resolve(intent, _snapshot(), set()) is None


def test_sysnav_resolver_rejects_unknown_object_uid() -> None:
    intent = NavigationIntent(mode=MotionGoalMode.GO_TO_OBJECT, target_object_uid="missing")

    with pytest.raises(ValueError, match="no target object"):
        SysNavGoalResolver(FakeSysNavProvider(_snapshot().viewpoints[0])).resolve(
            intent, _snapshot(), set()
        )


def test_snapshot_provider_intersects_target_and_anchor_visibility() -> None:
    snapshot = _snapshot()
    snapshot = SemanticMapSnapshot(
        timestamp=snapshot.timestamp,
        robot_pose=snapshot.robot_pose,
        objects=(
            snapshot.objects[0],
            ObjectNodeSnapshot(
                uid="shelf-1",
                label="shelf",
                position=(8.0, 8.0, 0.0),
                visible_viewpoints=("vp-2",),
            ),
        ),
        viewpoints=(
            ViewpointSnapshot(
                uid="vp-1",
                pose=snapshot.viewpoints[0].pose,
                visible_objects=("book-1",),
            ),
            ViewpointSnapshot(
                uid="vp-2",
                pose=snapshot.viewpoints[1].pose,
                visible_objects=("book-1", "shelf-1"),
            ),
        ),
    )
    intent = NavigationIntent(
        mode=MotionGoalMode.VERIFY_RELATION,
        target_object_uid="book-1",
        anchor_object_uid="shelf-1",
        relation_edge_id="book-1:shelf-1:on",
    )

    selected = SnapshotViewpointProvider().select_viewpoint(intent, snapshot, set())

    assert selected is not None
    assert selected.uid == "vp-2"


def test_sysnav_resolver_rejects_provider_returning_excluded_viewpoint() -> None:
    intent = NavigationIntent(mode=MotionGoalMode.GO_TO_OBJECT, target_object_uid="book-1")
    with pytest.raises(ValueError, match="excluded viewpoint"):
        SysNavGoalResolver(NonFilteringProvider(_snapshot().viewpoints[0])).resolve(
            intent, _snapshot(), {"vp-1"}
        )


def test_pre_resolved_resolver_does_not_invent_pose() -> None:
    intent = NavigationIntent(mode=MotionGoalMode.GO_TO_OBJECT, target_object_uid="book-1")

    assert PreResolvedGoalResolver().resolve(intent, _snapshot(), set()) is None
