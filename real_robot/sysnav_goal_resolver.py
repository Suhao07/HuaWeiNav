"""Resolve VLN semantic intents into SysNav-owned viewpoint goals.

The module deliberately contains no object-category rules and no local path
planner.  SysNav owns viewpoint generation, visibility, collision checking,
and path selection; this module only adapts the selected viewpoint to the
platform-neutral ``MotionGoal`` contract.
"""

from __future__ import annotations

from typing import Optional, Protocol, Set

from real_robot.contracts import (
    MotionGoal,
    MotionGoalMode,
    NavigationIntent,
    SemanticMapSnapshot,
    ViewpointSnapshot,
)


class ViewpointProvider(Protocol):
    """Provider boundary for a SysNav-selected executable viewpoint."""

    def select_viewpoint(
        self,
        intent: NavigationIntent,
        snapshot: SemanticMapSnapshot,
        excluded_viewpoint_ids: Set[str],
    ) -> Optional[ViewpointSnapshot]:
        """Return one SysNav-selected viewpoint, or ``None`` when unavailable.

        Args:
            intent: Semantic target, anchor, relation, and view objective.
            snapshot: Latest read-only semantic map snapshot.
            excluded_viewpoint_ids: Viewpoints already attempted for the
                current semantic candidate.

        Returns:
            A viewpoint selected by SysNav, or ``None`` when no candidate is
            currently available.
        """


class SnapshotViewpointProvider:
    """Read SysNav-ordered viewpoints already present in a map snapshot.

    This provider performs only identity intersection. It does not score
    distance, visibility, or collision. The snapshot order is treated as the
    order supplied by the lower SysNav viewpoint manager.
    """

    def select_viewpoint(
        self,
        intent: NavigationIntent,
        snapshot: SemanticMapSnapshot,
        excluded_viewpoint_ids: Set[str],
    ) -> Optional[ViewpointSnapshot]:
        """Return the first unattempted viewpoint matching the intent.

        Args:
            intent: Semantic target, anchor, or relation intent.
            snapshot: Snapshot containing SysNav viewpoint records.
            excluded_viewpoint_ids: Previously attempted viewpoint IDs.

        Returns:
            The first matching viewpoint in provider order, or ``None``.
        """

        requested_uid = str(intent.metadata.get("viewpoint_uid", "") or "").strip()
        viewpoints = {str(item.uid): item for item in snapshot.viewpoints}
        if requested_uid:
            item = viewpoints.get(requested_uid)
            return None if item is None or requested_uid in excluded_viewpoint_ids else item

        visible_ids: Optional[set[str]] = None
        for object_uid in (intent.target_object_uid, intent.anchor_object_uid):
            if not object_uid:
                continue
            obj = snapshot.object_by_uid(object_uid)
            object_ids = set(str(value) for value in (obj.visible_viewpoints if obj else ()))
            visible_ids = object_ids if visible_ids is None else visible_ids & object_ids
        if visible_ids is None:
            visible_ids = set(viewpoints)

        for item in snapshot.viewpoints:
            item_uid = str(item.uid)
            if item_uid in visible_ids and item_uid not in excluded_viewpoint_ids:
                return item
        return None


class GoalResolver(Protocol):
    """Minimal boundary between semantic policy and motion execution."""

    def resolve(
        self,
        intent: NavigationIntent,
        snapshot: SemanticMapSnapshot,
        excluded_viewpoint_ids: Set[str],
    ) -> Optional[MotionGoal]:
        """Convert one semantic intent into an executable motion goal.

        Args:
            intent: Semantic target, anchor, relation, and view objective.
            snapshot: Latest read-only semantic map snapshot.
            excluded_viewpoint_ids: Viewpoints already attempted for the
                current semantic candidate.

        Returns:
            A platform-neutral motion goal, or ``None`` when no executable
            SysNav viewpoint is available.
        """


class PreResolvedGoalResolver:
    """Pass through goals that were already resolved by a non-semantic test policy.

    This adapter exists for smoke policies and existing offline tests.  It does
    not derive a pose from an object or room and therefore cannot reintroduce
    the old centroid fallback into the semantic runtime.
    """

    def resolve(
        self,
        intent: NavigationIntent,
        snapshot: SemanticMapSnapshot,
        excluded_viewpoint_ids: Set[str],
    ) -> Optional[MotionGoal]:
        """Return the intent's explicit pose when one is already available.

        Args:
            intent: Intent produced by a test or pre-resolved policy.
            snapshot: Unused semantic snapshot retained for interface parity.
            excluded_viewpoint_ids: Unused attempt set retained for parity.

        Returns:
            Motion goal when ``intent.goal_pose`` exists; otherwise ``None``.
        """

        del snapshot, excluded_viewpoint_ids
        if intent.goal_pose is None:
            return None
        return intent.to_motion_goal()


class SysNavGoalResolver:
    """Adapt a SysNav-selected viewpoint to a VLN ``MotionGoal``.

    The provider is intentionally injected because the current migrated ROS
    workspace does not yet expose SysNav viewpoint poses.  In deployment it
    will be backed by a SysNav bridge; tests can provide an in-memory provider.
    This class never ranks viewpoints or computes a path.
    """

    def __init__(self, provider: ViewpointProvider) -> None:
        """Initialize the resolver.

        Args:
            provider: SysNav-owned viewpoint selection provider.

        Raises:
            TypeError: If no provider is supplied.
        """

        if provider is None:
            raise TypeError("SysNavGoalResolver requires a viewpoint provider")
        self.provider = provider

    def resolve(
        self,
        intent: NavigationIntent,
        snapshot: SemanticMapSnapshot,
        excluded_viewpoint_ids: Set[str],
    ) -> Optional[MotionGoal]:
        """Resolve one semantic intent without inventing a robot pose.

        Args:
            intent: Semantic target/anchor/relation intent from VLN policy.
            snapshot: Latest semantic map snapshot.
            excluded_viewpoint_ids: Viewpoint IDs already tried for this task.

        Returns:
            A motion goal containing the SysNav-selected pose, or ``None`` if
            SysNav has no executable viewpoint at this time.

        Raises:
            ValueError: If the provider returns an excluded viewpoint or a
                viewpoint without a usable frame, or the intent refers to an
                object UID absent from the snapshot.
        """

        if intent.mode in {MotionGoalMode.WAIT, MotionGoalMode.STOP}:
            return intent.to_motion_goal()

        for role, object_uid in (
            ("target", intent.target_object_uid),
            ("anchor", intent.anchor_object_uid),
        ):
            if object_uid and snapshot.object_by_uid(object_uid) is None:
                raise ValueError(f"SysNav snapshot has no {role} object {object_uid!r}")

        viewpoint = self.provider.select_viewpoint(intent, snapshot, set(excluded_viewpoint_ids))
        if viewpoint is None:
            return None
        viewpoint_id = str(viewpoint.uid or "").strip()
        if not viewpoint_id:
            raise ValueError("SysNav viewpoint must have a stable uid")
        if viewpoint_id in excluded_viewpoint_ids:
            raise ValueError(f"SysNav provider returned excluded viewpoint {viewpoint_id!r}")
        if viewpoint.pose is None or not viewpoint.pose.frame_id:
            raise ValueError(f"SysNav viewpoint {viewpoint_id!r} has no frame_id")

        # 中文核心边界：语义模块决定“看什么”，SysNav 决定“站在哪里”；
        # 这里仅把已选中的 viewpoint pose 传递给运动层，不重新规划路径。
        metadata = {
            **dict(intent.metadata or {}),
            "viewpoint_uid": viewpoint_id,
            "viewpoint_source": "sysnav",
            "viewpoint_room_id": viewpoint.room_id,
            "viewpoint_visible_objects": list(viewpoint.visible_objects),
        }
        return MotionGoal(
            mode=intent.mode,
            goal_pose=viewpoint.pose,
            target_object_uid=intent.target_object_uid,
            anchor_object_uid=intent.anchor_object_uid,
            relation_edge_id=intent.relation_edge_id,
            reason=intent.reason,
            metadata=metadata,
        )
