"""Backend-independent baseline decisions and execution lifecycle."""
from dataclasses import asdict, dataclass, field
from hashlib import sha256
import math

import numpy as np


@dataclass(frozen=True)
class GuidedVerification:
    """任务归约结果；STOP 由执行层负责，参照物搜索始终独立于任务完成。"""

    candidate_uid: str
    role: str
    target_id: str
    decision: str
    complete: bool
    pose_gap: dict = field(default_factory=dict)
    relation_outcomes: list[dict] = field(default_factory=list)


class GuidedPolicy:
    """共享候选选择与恢复策略；后端负责观测、运动和任务归约。"""

    def initialize_state(self, *, deferred_candidates, review_interval):
        """Initialize revision-scoped histories without owning backend state."""
        self.sequence = 0
        self.recovery_context = {}
        self.deferred_candidates = deferred_candidates
        self.deferred_evidence = {}
        self.candidate_view_states = {}
        self.candidate_motion_goals = {}
        self.action_epochs = {}
        self.relation_outcomes = {}
        self.deferred_context = {}
        self.observed_views = {}
        self.search_review_step = self.step
        self.search_review_interval = review_interval

    @staticmethod
    def candidate_evidence(obj):
        """Fingerprint only instance crop content; missing evidence stays unknown."""
        if obj.rgb is None or obj.bbox is None:
            return None
        image = np.asarray(obj.rgb)
        x1, y1, x2, y2 = np.asarray(obj.bbox).reshape(-1)[:4].astype(int)
        height, width = image.shape[:2]
        crop = image[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
        if crop.size == 0:
            return None
        digest = sha256(str((crop.shape, crop.dtype)).encode())
        digest.update(crop.tobytes())
        return digest.hexdigest()

    def action_epoch(self, candidate_uid):
        return self.action_epochs.get((self.revision, candidate_uid), 0)

    def attempted_positions(self, candidate_uid):
        key = (self.revision, candidate_uid, self.action_epoch(candidate_uid))
        return self.candidate_motion_goals.get(key, {})

    def view_attempted(self, candidate_uid, position):
        """Match prior views within the controller's XY arrival tolerance."""
        return any(math.dist(position[:2], old[:2]) <= self.arrival_tolerance
                   for old in self.attempted_positions(candidate_uid))

    def motion_finished(self, feedback):
        """Retain terminal execution facts; submission alone consumes no view."""
        uid = feedback.intent.target_object_uid
        if uid is None:
            return
        metadata = feedback.intent.metadata
        key = (metadata["task_revision"], uid, metadata["candidate_action_epoch"])
        pose = tuple(float(x) for x in feedback.intent.goal_pose.position)
        self.candidate_motion_goals.setdefault(key, {})[pose] = feedback.status.status.value

    def unresolved_relations(self, uid):
        outcomes = self.relation_outcomes.get((self.revision, uid), [])
        verified = {row["predicate_id"] for row in outcomes if row["status"] == "verified"}
        return [row for row in outcomes if row["predicate_id"] not in verified]

    def relation_anchor_uids(self, uid, outcome):
        from instruction_adapter.concept_matcher import _concept_signature
        from instruction_adapter.ontology import normalize_term
        from instruction_adapter.verifier import instruction_hash

        session = self.tools.session
        concept = next(t.concept_query() for t in session.plan.targets
                       if t.id == outcome["anchor_concept_id"])
        terms = {normalize_term(term) for term in concept.match_terms}
        anchors = set()
        for obj in session.snapshot.objects:
            if obj.uid == uid:
                continue
            label = normalize_term(obj.label)
            match = session.concept_matcher.records.get((
                instruction_hash(session.spec.raw_instruction), _concept_signature(concept), obj.uid, label))
            if (match is not None and match.candidate_eligible) or (match is None and label in terms):
                anchors.add(obj.uid)
        if outcome["anchor_uid"]:
            anchors.add(outcome["anchor_uid"])
        return anchors

    def observation_context(self, uid):
        session = self.tools.session
        pending = self.unresolved_relations(uid)
        tokens = set()
        groups = [(uid,)] if not pending else []
        for row in pending:
            anchors = self.relation_anchor_uids(uid, row)
            tokens.update((row["predicate_id"], anchor) for anchor in anchors)
            groups.extend((uid, anchor) for anchor in anchors)
        for group in groups:
            views = self.observed_views.setdefault((self.revision, *group), [])
            for image in session.snapshot.metadata.get("observation_images", []):
                if not all(item in image.get("object_boxes", {})
                           and item in image.get("observed_object_uids", []) for item in group):
                    continue
                position = np.asarray(image.get("position", []), dtype=float)
                rotation = np.asarray(image.get("rotation", []), dtype=float)
                if position.shape != (3,) or rotation.shape != (3, 3):
                    continue
                if not np.isfinite(position).all() or not np.isfinite(rotation).all():
                    continue
                # SO(3) 相对转角满足 tr(RᵀR_old) = 1 + 2cosθ；像素变化本身不构成新视角。
                if not any(np.linalg.norm(position - old_position) <= self.arrival_tolerance
                           and np.trace(rotation.T @ old_rotation) >= 1 + 2 * math.cos(math.radians(15))
                           for old_position, old_rotation in views):
                    views.append((position, rotation))
            tokens.update(("view", *group, index) for index in range(len(views)))
        return tokens

    def defer_candidate(self, reason):
        uid = self.candidate_uid
        self.deferred_candidates.add(uid)
        fingerprint = self.evidence_fingerprint(uid)
        key = (self.revision, uid)
        self.deferred_evidence.setdefault(key, set()).add(fingerprint)
        self.deferred_context.setdefault(key, set()).update(self.observation_context(uid))
        self.candidate_view_states[key] = self.view_state
        self.record("candidate_deferred", candidate_uid=uid, reason=reason,
                    evidence_fingerprint=fingerprint)

    def observation_updated(self):
        self.refresh_scene()
        for uid in set(self.deferred_candidates) | {key[1] for key in self.candidate_view_states
                                                  if key[0] == self.revision}:
            context = self.observation_context(uid)
            key = (self.revision, uid)
            if uid in self.deferred_candidates and context.difference(self.deferred_context.get(key, set())):
                self.deferred_context.setdefault(key, set()).update(context)
                self.action_epochs[key] = self.action_epoch(uid) + 1
                self.deferred_candidates.remove(uid)
                self.record("candidate_evidence_updated", candidate_uid=uid,
                            evidence_fingerprint=self.evidence_fingerprint(uid),
                            action_epoch=self.action_epoch(uid))
        self.observation_refreshed()

    def observation_refreshed(self):
        """Commit backend observation effects after shared evidence refresh succeeds."""

    def select(self, episode_idx=None):
        """Prefer an untried candidate view, then pending scans, then exploration."""
        uid = self.select_candidate(episode_idx)
        while uid is not None:
            views = self.initial_views(uid)
            available = [v for v in views if not self.view_attempted(uid, v.pose.position)]
            if available:
                view = next((v for v in available if v.metadata.get("recommended")), available[0])
                self.submit(view.pose.position, candidate_uid=uid, source="baseline_viewpoint")
                return True
            self.defer_candidate("available_views_already_attempted")
            uid = self.select_candidate(episode_idx)
        if not self.scan_complete:
            self.observe_pending(episode_idx)
            self.observation_updated()
            return True
        node, source = self.select_exploration(episode_idx)
        if node is None:
            self.record("search_exhausted", decision_source=source)
            return False
        node, source = self.review_search_branch(node, source, episode_idx)
        if source in {"interpretation_changed", "decision_reselect"}:
            return True
        if source == "agent_search_review":
            with self.runtime.choice_guard():
                self.submit(node.position, node=node, source=source)
        else:
            self.submit(node.position, node=node, source=source)
        return True

    def review_search_branch(self, node, source, episode_idx):
        """Review periodically without resetting the clock on candidate discovery."""
        if (self.step - self.search_review_step < self.search_review_interval
                or self.runtime.model_calls >= self.runtime.max_model_calls):
            return node, source
        node, source, rows = self.review_options(node, source, episode_idx)
        options = [{**row, "frontier_uid": f"node:{row['node_idx']}"} for row in rows]
        if not options:
            return node, source
        recommended = f"node:{node.idx}"
        revision = self.revision
        selected = self.runtime.choose_frontier(options, recommended, {
            "reason": "prolonged_target_search", "steps_since_review": self.step - self.search_review_step,
        })
        self.search_review_step = self.step
        if selected is None:
            event = "interpretation_changed" if self.revision != revision else "decision_reselect"
            self.record(event, decision_source="agent_search_review", task_revision=self.revision)
            return None, event
        index = next(row["node_idx"] for row in options if row["frontier_uid"] == selected)
        self.record("search_review", decision_source="agent", baseline_uid=recommended,
                    selected_uid=selected, options=options)
        return self.node_for_index(index), "agent_search_review"

    def verify(self, scored_position):
        """Reduce captured evidence to completion, another view, or recovery."""
        result = self.capture_verification(scored_position)
        if result is None:
            return "episode_limit"
        if result.role == "anchor":
            session = self.tools.session
            status = session.anchor_search_ledger.get(
                session.spec.raw_instruction, result.target_id, result.candidate_uid).status
            self.record("anchor_searched", anchor_uid=result.candidate_uid, concept_id=result.target_id,
                        status=status, pose_gap=result.pose_gap)
            if not self.restore_camera():
                return "episode_limit"
            if self.review_due():
                self.recovery_context = {
                    "reason": "prolonged_target_search", "anchor_uid": result.candidate_uid,
                    "concept_id": result.target_id, "anchor_status": status,
                    "steps_since_review": self.step - self.search_review_step}
                return "recovery"
            return "select"
        self.record("verified", candidate_uid=result.candidate_uid, decision=result.decision,
                    task_complete=result.complete, pose_gap=result.pose_gap)
        if result.complete:
            return "verified_finish"
        if not self.restore_camera():
            return "episode_limit"
        # 未进行关系复核的补拍结果不能清除已有关系缺口。
        if result.relation_outcomes or result.decision != "need_better_view":
            self.relation_outcomes[(self.revision, result.candidate_uid)] = result.relation_outcomes
        if result.decision == "reject_candidate":
            self.defer_candidate(result.decision)
            return "select"
        if result.decision in {"need_relation_check", "reject_relation"}:
            self.recovery_context = {"reason": "relation_evidence_gap", "candidate_uid": result.candidate_uid,
                                     "decision": result.decision, "relation_outcomes": result.relation_outcomes}
            if self.relation_search_required(result.candidate_uid):
                self.defer_candidate(result.decision)
                return "select"
        if self.review_due():
            self.recovery_context = {
                "reason": "prolonged_unverified_candidate", "candidate_uid": result.candidate_uid,
                "decision": result.decision, "relation_outcomes": result.relation_outcomes,
                "steps_since_review": self.step - self.search_review_step}
            return "recovery"
        if self.continue_view_control() or any(row["status"] == "insufficient_evidence"
                for row in self.unresolved_relations(result.candidate_uid)):
            available = [v for v in self.feedback_views(result.candidate_uid)
                         if not self.view_attempted(result.candidate_uid, v.pose.position)]
            if available:
                view = next((v for v in available if v.metadata.get("recommended")), available[0])
                self.submit(view.pose.position, candidate_uid=result.candidate_uid, source="baseline_view_feedback")
                return "moving"
            self.view_state.exhausted = True
        if self.view_state.active and self.view_state.budget_exhausted():
            self.defer_candidate("view_options_exhausted")
            self.recovery_context = {"reason": "view_options_exhausted", "candidate_uid": result.candidate_uid,
                                     "view_objective": self.view_state.objective}
            return "recovery"
        self.defer_candidate(result.decision)
        return "select"

    def relation_search_required(self, uid):
        outcomes = self.relation_outcomes.get((self.revision, uid), [])
        if not outcomes:
            return True
        for predicate in {row["predicate_id"] for row in outcomes}:
            statuses = {row["status"] for row in outcomes if row["predicate_id"] == predicate}
            if statuses <= {"missing_anchor", "rejected_relation"}:
                return True
        return False

    def review_due(self):
        """Check the periodic interval and remaining model-call budget."""
        return (self.step - self.search_review_step >= self.search_review_interval
                and self.runtime.model_calls < self.runtime.max_model_calls)

    def recover(self, reason, episode_idx=None):
        """Choose one untried executable action, or reselect after a task revision."""
        self.refresh_scene()
        _, _, rows = self.exploration_options(episode_idx)
        failed_index = reason.get("goal", {}).get("node_index")
        options = [{**row, "action_uid": f"frontier:{row['node_idx']}", "kind": "frontier"}
                   for row in rows if row["node_idx"] != failed_index]
        uid = reason.get("candidate_uid")
        if uid is not None and uid not in self.deferred_candidates:
            relation_gap = any(row["status"] == "insufficient_evidence"
                               for row in self.unresolved_relations(uid))
            views = ([asdict(view) for view in self.feedback_views(uid)] if relation_gap
                     else self.tools.query_viewpoints(uid))
            for view in views:
                if not self.view_attempted(uid, view["pose"]["position"]):
                    options.append({**view, "action_uid": view["uid"], "kind": "viewpoint", "candidate_uid": uid})
        if not options:
            if uid is not None and self.unresolved_relations(uid):
                if uid not in self.deferred_candidates:
                    self.defer_candidate("relation_views_exhausted")
                return "select"
            return "select" if reason.get("anchor_uid") is not None else "search_exhausted"
        if self.runtime.model_calls >= self.runtime.max_model_calls:
            return "decision_budget_exhausted"
        revision = self.revision
        selected = self.runtime.choose_recovery(options, reason)
        self.search_review_step = self.step
        if selected is None:
            event = "interpretation_changed" if self.revision != revision else "decision_reselect"
            self.record(event, decision_source="agent_recovery", task_revision=self.revision)
            return "select"
        self.record("agent_intervention", reason=reason, decision_source="agent_recovery",
                    selected_uid=selected, options=options)
        choice = next(row for row in options if row["action_uid"] == selected)
        with self.runtime.choice_guard():
            if choice["kind"] == "viewpoint":
                self.submit(choice["pose"]["position"], candidate_uid=uid, source="agent_recovery")
            else:
                node = self.node_for_index(choice["node_idx"])
                self.submit(node.position, node=node, source="agent_recovery")
        return "moving"


def run_guided_loop(guided, execution):
    """观测更新后再标记前沿；已接受的运动完成后才检查决策预算。"""
    motion = guided.tools.motion
    loop_error = None
    try:
        execution.initialize()
        state = "select"
        while True:
            reason = execution.limit_reason()
            if reason == "decision_budget_exhausted" and motion.active_request_id is not None:
                reason = None
            if reason is not None:
                return reason
            if motion.active_request_id is not None:
                execution.advance()
                feedback = motion.poll_active()
                execution.after_motion(feedback)
                if feedback.tracking_active:
                    continue
                guided.motion_finished(feedback)
                metadata = feedback.intent.metadata
                guided.record("motion_terminal", status=feedback.status.status.value,
                              candidate_uid=feedback.intent.target_object_uid,
                              position=feedback.intent.goal_pose.position,
                              task_revision=metadata["task_revision"], metadata=metadata)
                if not feedback.status.succeeded():
                    guided.recovery_context = {
                        "reason": "motion_terminal_failure", "status": feedback.status.status.value,
                        "candidate_uid": feedback.intent.target_object_uid, "goal": metadata}
                    state = "recovery"
                    continue
                if metadata.get("node_index") is not None or metadata.get("frontier_uid") is not None:
                    execution.frontier_arrived(feedback)
                    guided.observation_updated()
                    execution.frontier_observed(feedback)
                    state = "select"
                else:
                    execution.object_arrived(feedback)
                    state = guided.verify(feedback.intent.goal_pose.position)
            elif state == "select":
                if not guided.select(execution.episode_idx):
                    return "search_exhausted"
            elif state == "recovery":
                state = guided.recover(guided.recovery_context, execution.episode_idx)
            if state in {"verified_finish", "search_exhausted", "decision_budget_exhausted", "episode_limit"}:
                if state == "verified_finish":
                    execution.finish_success()
                return state
    except BaseException as error:
        loop_error = error
        raise
    finally:
        if motion.active_request_id is not None:
            try:
                motion.cancel_and_wait()
            except BaseException as cleanup_error:
                if loop_error is not None:
                    raise loop_error from cleanup_error
                raise
