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
    session.constraint_evaluator.register_mapper_objects(mapper=session, step=step)
    result = query_candidates(
        session.plan, session.progress, session.snapshot, session.verification_ledger,
        matcher=session.concept_matcher, anchor_ledger=session.anchor_search_ledger,
        step=step, prior_adapter=session.prior_map_policy_adapter,
        prior_result=session.prior_result,
    )
    demands = {target.id for target in session.spec.targets if target.demand}
    session.destination_hypotheses = [
        {"target_id": candidate.target_id, "candidate_uid": candidate.obj.uid,
         "facility_concept": candidate.obj.label, "status": "requires_instance_verification",
         "task_revision": session.spec.revision, "observation_timestamp": session.snapshot.timestamp,
         "reason": candidate.record.get("reason", "")}
        for candidate in result.candidates if candidate.target_id in demands
    ]
    return result


class ToolProposalError(ValueError):
    """An inadmissible agent proposal, not a backend execution failure."""


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
        self.movement_budget_remaining = None
        self._finish_verification = None

    def observe(self):
        """Project task state, scoped stored judgments and backend observations.

        Only snapshot.metadata['observation_images'] publishes current camera
        frames. No capture, verification or historical-object image fallback is
        performed; the returned image metadata is detached from the snapshot.
        """
        from copy import deepcopy
        from dataclasses import asdict
        from verification.store import instruction_hash
        snapshot = self.session.snapshot
        spec = self.session.spec
        task_hash = instruction_hash(spec.raw_instruction)
        return {
            "raw_instruction": spec.raw_instruction,
            "semantic_authority": "raw_instruction",
            "structured_interpretation_role": "revisable_execution_aid",
            "task": spec.as_dict(),
            "progress": self.session.progress.as_dict(),
            "completion_review": self.propose_finish(),
            "movement_budget_remaining": self.movement_budget_remaining,
            "destination_hypotheses": deepcopy(self.session.destination_hypotheses),
            "search_context": {
                "revision": self.session.search_revision,
                "concepts": list({concept.id: concept.as_dict()
                                  for concept in self.session.search_concepts}.values()),
                "hypotheses": deepcopy(list(self.session.search_hypotheses.values())),
            },
            # Stored judgments retain their predicate/pair/dependency scope; no
            # query or new verification is performed to fill missing evidence.
            "verification_records": [record.as_dict()
                                     for record in self.session.verification_ledger.records.values()
                                     if record.instruction_hash == task_hash
                                     and record.task_revision == spec.revision],
            "observation_images": deepcopy(snapshot.metadata.get("observation_images", [])),
            "timestamp": snapshot.timestamp,
            "robot_pose": snapshot.robot_pose.as_dict(),
            # 地图按发现顺序追加对象；截取前缀会永久隐藏后来发现的目标。
            "objects": [obj.concept_payload() for obj in snapshot.objects],
            "frontiers": [asdict(item) for item in snapshot.frontiers],
            "motion": {"phase": self.motion.phase, "held": self.motion.held,
                       "control_epoch": self.motion.control_epoch,
                       "active_request_id": self.motion.active_request_id},
        }

    def install_interpretation(self, plan, *, task_revision, search_revision, snapshot, record):
        with self.motion.idle_guard(control_epoch=record["control_epoch"]):
            return self.session.adopt_interpretation(
                plan, task_revision=task_revision, search_revision=search_revision,
                snapshot=snapshot, record=record)

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
        with self.motion.submission_guard():
            if self.movement_budget_remaining is not None and self.movement_budget_remaining <= 0:
                return {"accepted": False, "reason": "movement_budget_exhausted"}
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
                self._finish_verification = None
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
            self._finish_verification = None
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
        epoch = self.motion.control_epoch
        with self.motion.idle_guard(control_epoch=epoch):
            self._finish_verification = None
        evidence = dict(self.evidence_provider(candidate_uid))
        captured_snapshot = self.session.snapshot
        captured_task = self.session.spec.as_dict()
        captured_progress = self.session.progress.as_dict()
        from copy import deepcopy
        captured_observations = deepcopy(captured_snapshot.metadata.get("observation_images", []))
        evidence["dependencies"] = {
            **evidence.get("dependencies", {}),
            **object_dependencies(source_snapshot, candidate_uid),
        }
        evidence["target_id"] = target.id
        evidence["task_progress"] = self.session.progress.as_dict()
        evidence["prior_verifications"] = self.observe()["verification_records"]
        candidate = candidate_from_object(obj, canonical_label=target.name)
        result = self.verifier.verify(
            raw_instruction=self.session.spec.raw_instruction,
            instruction_plan=self.session.plan, candidate=candidate, evidence=evidence,
        )
        if object_dependencies(source_snapshot, candidate_uid, result.anchor_uid) != object_dependencies(
            self.session.snapshot, candidate_uid, result.anchor_uid
        ):
            raise ValueError("Verification observation dependencies changed during inference")
        with self.motion.idle_guard(control_epoch=epoch):
            if captured_task != self.session.spec.as_dict() or task_revision != self.session.spec.revision:
                raise ValueError("Verification task changed during inference")
            current = self.session.snapshot
            if (captured_snapshot.timestamp != current.timestamp or captured_snapshot.robot_pose != current.robot_pose
                    or captured_observations != current.metadata.get("observation_images", [])
                    or captured_progress != self.session.progress.as_dict()):
                raise ValueError("Verification stop context changed during inference")
            self.session.verification_ledger.put(self.session.spec.raw_instruction, candidate_uid, result)
            apply_verification(self.session.progress, target, candidate, result, evidence)
            self._finish_verification = {
                "result": result,
                "task": self.session.spec.as_dict(),
                "progress": self.session.progress.as_dict(),
                "timestamp": captured_snapshot.timestamp,
                "pose": captured_snapshot.robot_pose,
                "observation_images": captured_observations,
                "target_id": target.id,
                "control_epoch": epoch,
                "object_dependencies": object_dependencies(self.session.snapshot, candidate_uid, result.anchor_uid),
            }
        return result.as_dict()

    def propose_finish(self):
        """Require current full-instruction approval, counted progress and idle motion."""
        from scene.dependencies import object_dependencies
        from verification.images import image_dependencies_valid
        receipt = self._finish_verification
        semantic_complete = False
        review = {}
        if receipt is not None:
            result = receipt["result"]
            review = result.diagnostics.get("original_instruction_review", {})
            semantic_complete = bool(
                result.accepted and review.get("task_satisfied") is True
                and review.get("coverage_complete") is True
                and review.get("interpretation_consistent") is True
                and result.task_revision == self.session.spec.revision
                and receipt["task"] == self.session.spec.as_dict()
                and receipt["progress"] == self.session.progress.as_dict()
                and receipt["timestamp"] == self.session.snapshot.timestamp
                and receipt["pose"] == self.session.snapshot.robot_pose
                and receipt["observation_images"] == self.session.snapshot.metadata.get("observation_images", [])
                and receipt["control_epoch"] == self.motion.control_epoch
                and receipt["object_dependencies"] == object_dependencies(
                    self.session.snapshot, result.target_uid, result.anchor_uid)
                and image_dependencies_valid(result.dependencies)
            )
        return {"accepted": self.session.progress.completed and semantic_complete
                and self.motion.active_request_id is None and not self.motion.held,
                "task_complete": self.session.progress.completed,
                "original_instruction_satisfied": semantic_complete,
                "original_instruction_review": review,
                "verified_target_id": receipt["target_id"] if semantic_complete else None,
                "motion_phase": self.motion.phase}

    def dispatch(self, name, arguments):
        """Reject invalid proposals before invoking any backend operation."""
        if name not in self.names:
            raise ToolProposalError(f"Unknown navigation tool: {name}")
        snapshot = self.session.snapshot
        if name in {"submit_motion", "verify_target"}:
            if arguments["task_revision"] != self.session.spec.revision:
                raise ToolProposalError("Proposal belongs to another task revision")
            if self.motion.held:
                raise ToolProposalError("Motion is held; execution proposals are not admitted")
            if self.motion.active_request_id is not None and (
                    name != "submit_motion" or arguments["request_id"] != self.motion.active_request_id):
                raise ToolProposalError("Wait for terminal motion feedback before another execution proposal")
        if name == "submit_motion":
            if arguments["control_epoch"] != self.motion.control_epoch:
                raise ToolProposalError("Motion proposal belongs to another control epoch")
            if arguments.get("frontier_uid") is not None:
                if arguments.get("candidate_uid") is not None or arguments.get("viewpoint_uid") is not None:
                    raise ToolProposalError("Choose either a frontier or an object viewpoint")
                if not any(item.uid == arguments["frontier_uid"] for item in snapshot.frontiers):
                    raise ToolProposalError("Unknown observed frontier")
            else:
                uid = arguments.get("candidate_uid")
                obj = snapshot.object_by_uid(uid)
                if obj is None:
                    raise ToolProposalError(f"Unknown observed instance: {uid}")
                if not any(view.uid == arguments.get("viewpoint_uid") and (
                        view.uid in obj.visible_viewpoints or uid in view.visible_objects) for view in snapshot.viewpoints):
                    raise ToolProposalError("Viewpoint is not associated with the observed instance")
        if name in {"query_viewpoints", "verify_target"}:
            uid = arguments["candidate_uid"]
            if snapshot.object_by_uid(uid) is None:
                raise ToolProposalError(f"Unknown observed instance: {uid}")
        if name == "verify_target" and not any(
                target.id == arguments["target_id"] and target.terminal for target in self.session.spec.targets):
            raise ToolProposalError("Unknown terminal target in the current interpretation")
        return getattr(self, name)(**arguments)
