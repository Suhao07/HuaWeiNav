"""工具绑定当前任务的服务，领域算法保持唯一实现。"""
from __future__ import annotations
from scene.query import CandidateSet, query_candidates
from task.session import TaskSession


def query_scene(session: TaskSession, *, step: int | None = None) -> CandidateSet:
    """Query observed candidates using the current task's evidence and priors.

    Args:
        session: Current task revision, observed snapshot and domain services.
        step: Observation sequence retained in matching diagnostics.

    Returns:
        Ranked observed candidates and matching/exclusion evidence. Querying
        leaves task progress and motion ownership unchanged.
    """
    return query_candidates(
        session.plan, session.progress, session.snapshot, session.verification_ledger,
        matcher=session.concept_matcher, anchor_ledger=session.anchor_search_ledger,
        step=step, prior_adapter=session.prior_map_policy_adapter,
        prior_result=session.prior_result,
    )


class NavigationTools:
    """Bind the eight Agent tools to task, evidence and motion owners.

    Args:
        session: Active task and observed scene.
        motion: The runtime's existing MotionSupervisor.
        verifier: Existing final evidence verifier.
        evidence_provider: Backend callable taking a candidate UID and returning
            current verifier evidence. Arrival barriers belong to this provider.

    Agent 提交的是观测到的实例和视点 ID；坐标从后端快照读取。
    """
    names = (
        "observe", "query_scene", "query_viewpoints", "submit_motion",
        "motion_status", "cancel_motion", "verify_target", "propose_finish",
    )

    def __init__(self, session, motion, verifier, evidence_provider):
        self.session = session
        self.motion = motion
        self.verifier = verifier
        self.evidence_provider = evidence_provider
        self.last_goal_id = None

    def observe(self):
        """Project bounded scene facts and authoritative state into model input."""
        from dataclasses import asdict
        snapshot = self.session.snapshot
        return {
            "task": self.session.spec.as_dict(),
            "progress": self.session.progress.as_dict(),
            "timestamp": snapshot.timestamp,
            "robot_pose": snapshot.robot_pose.as_dict(),
            "objects": [obj.concept_payload() for obj in snapshot.objects[:32]],
            "frontiers": [asdict(item) for item in snapshot.frontiers[:32]],
            "motion": {"phase": self.motion.phase, "held": self.motion.held,
                       "control_epoch": self.motion.control_epoch},
        }

    def query_scene(self):
        """Return domain-ranked candidates without choosing a motion."""
        result = query_scene(self.session)
        return {
            "candidates": [{"candidate_uid": c.obj.uid, "target_id": c.target_id,
                            "role": c.role, "label": c.canonical_label}
                           for c in result.candidates],
            "reason": result.reason,
            "excluded": result.skipped_objs,
        }

    def query_viewpoints(self, candidate_uid):
        """List observed views associated with this instance."""
        from dataclasses import asdict
        snapshot = self.session.snapshot
        obj = snapshot.object_by_uid(candidate_uid)
        if obj is None:
            raise ValueError(f"Unknown observed instance: {candidate_uid}")
        return [asdict(v) for v in snapshot.viewpoints
                if v.uid in obj.visible_viewpoints or candidate_uid in v.visible_objects]

    def submit_motion(self, task_revision, control_epoch, request_id,
                      candidate_uid=None, viewpoint_uid=None, frontier_uid=None):
        """Submit an observed viewpoint through the sole motion owner."""
        from motion.contracts import MotionGoalMode, NavigationIntent
        if task_revision != self.session.spec.revision:
            raise ValueError("Motion proposal belongs to another task revision")
        if frontier_uid is not None:
            from scene.contracts import Pose3D
            if candidate_uid is not None or viewpoint_uid is not None:
                raise ValueError("Choose either a frontier or an object viewpoint")
            frontier = next(f for f in self.session.snapshot.frontiers if f.uid == frontier_uid)
            intent = NavigationIntent(
                mode=MotionGoalMode.GO_TO_FRONTIER,
                goal_pose=Pose3D(frontier.position, frame_id=frontier.metadata["frame_id"]),
                metadata={"task_revision": task_revision, "frontier_uid": frontier_uid},
            )
            goal_id = self.motion.submit(intent, intent.to_motion_goal(), request_id=request_id,
                                         control_epoch=control_epoch)
            self.last_goal_id = goal_id
            return {"goal_id": goal_id, "phase": self.motion.phase}
        views = NavigationTools.query_viewpoints(self, candidate_uid)
        if viewpoint_uid not in {v["uid"] for v in views}:
            raise ValueError("Viewpoint is not associated with the observed instance")
        viewpoint = next(v for v in self.session.snapshot.viewpoints if v.uid == viewpoint_uid)
        intent = NavigationIntent(
            mode=MotionGoalMode.GO_TO_OBJECT, target_object_uid=candidate_uid,
            goal_pose=viewpoint.pose,
            metadata={"task_revision": task_revision, "viewpoint_uid": viewpoint_uid},
        )
        goal_id = self.motion.submit(intent, intent.to_motion_goal(), request_id=request_id,
                                     control_epoch=control_epoch)
        self.last_goal_id = goal_id
        return {"goal_id": goal_id, "phase": self.motion.phase}

    def motion_status(self):
        """Poll actual execution feedback; polling never submits another goal."""
        from dataclasses import asdict
        polled = self.motion.poll_active()
        status = polled.status if polled else (
            self.motion.controller.poll_status(self.last_goal_id) if self.last_goal_id is not None
            else self.motion.last_terminal_status
        )
        return {"phase": self.motion.phase,
                "status": asdict(status) if status is not None else None,
                "held": self.motion.held}

    def cancel_motion(self):
        """Request cancellation; motion_status provides terminal confirmation."""
        self.motion.cancel()
        return {"phase": self.motion.phase, "control_epoch": self.motion.control_epoch}

    def verify_target(self, candidate_uid, target_id, task_revision):
        """Verify backend evidence, store its judgment and apply the task reducer."""
        from instruction_adapter.verifier import candidate_from_object
        from task.reducer import apply_verification
        from scene.dependencies import object_dependencies
        if task_revision != self.session.spec.revision:
            raise ValueError("Verification request belongs to another task revision")
        target = next(t for t in self.session.spec.targets if t.id == target_id and t.terminal)
        obj = self.session.snapshot.object_by_uid(candidate_uid)
        if obj is None:
            raise ValueError(f"Unknown observed instance: {candidate_uid}")
        source_snapshot = self.session.snapshot
        evidence = dict(self.evidence_provider(candidate_uid))
        evidence["dependencies"] = {
            **evidence.get("dependencies", {}),
            **object_dependencies(source_snapshot, candidate_uid),
        }
        evidence["target_id"] = target.id
        candidate = candidate_from_object(obj, canonical_label=target.name)
        result = self.verifier.verify(
            raw_instruction=self.session.spec.raw_instruction,
            instruction_plan=self.session.plan, candidate=candidate, evidence=evidence,
        )
        if object_dependencies(source_snapshot, candidate_uid, result.anchor_uid) != object_dependencies(
            self.session.snapshot, candidate_uid, result.anchor_uid
        ):
            raise ValueError("Verification observation dependencies changed during inference")
        self.session.verification_ledger.put(self.session.spec.raw_instruction, candidate_uid, result)
        apply_verification(self.session.progress, target, candidate, result, evidence)
        return result.as_dict()

    def propose_finish(self):
        """Read domain completion and execution quiescence; never set progress."""
        return {"accepted": self.session.progress.completed
                and self.motion.active_request_id is None and not self.motion.held,
                "task_complete": self.session.progress.completed,
                "motion_phase": self.motion.phase}

    def dispatch(self, name, arguments):
        """Admit a named tool; argument and domain errors propagate to the caller."""
        if name not in self.names:
            raise ValueError(f"Unknown navigation tool: {name}")
        return getattr(self, name)(**arguments)
