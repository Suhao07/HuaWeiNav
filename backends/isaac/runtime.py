"""Run full-instruction Agent tool decisions against live Isaac observations and motion."""
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import time
import urllib.request

import numpy as np
from PIL import Image, ImageDraw

from backends.isaac.perception import IsaacObservationMap, visible_projection
from harness.runtime import AgentRuntime
from harness.tools import NavigationTools
from instruction.compiler import InitialInstructionCompiler
from instruction_adapter.compiler import compile_instruction_plan
from instruction_adapter.constraints import ConstraintEvaluator
from instruction_adapter.relation_verifier import DynamicRelationService
from instruction_adapter.verifier import FinalInstructionVerifier
from llm_utils.lvlm_call_tracker import counts, set_trace_dir
from motion.contracts import MotionGoalMode, NavigationStatus, NavigationStatusCode
from motion.supervisor import MotionSupervisor
from prior_map.observations import write_json
from scene.contracts import FrontierSnapshot, Pose3D, SemanticMapSnapshot, ViewpointSnapshot
from task.session import TaskSession
from verification.contracts import VerificationResult


BACKGROUND_CLASSES = ("chair", "table", "sofa", "bed", "cabinet", "shelf", "lamp",
                      "television", "door", "plant", "picture", "mirror", "book", "cup")


def pose_value(pose):
    """Convert the simulator's world-Z-up yaw to the shared base-pose contract."""
    yaw = float(pose["yaw"])
    return Pose3D(tuple(pose["position"]), (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)))


class IsaacClient:
    """Send bounded commands only to the run-owned loopback simulator."""

    def __init__(self, port):
        self.url = f"http://127.0.0.1:{int(port)}/command"

    def command(self, action, **arguments):
        """Return an execution receipt, propagating transport and simulator failures."""
        request = urllib.request.Request(self.url, data=json.dumps(
            {"action": action, **arguments}, allow_nan=False).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request, timeout=240) as response:
            return json.load(response)


class IsaacMotionController:
    """Advance camera motion only after MotionSupervisor admits an observed goal."""

    def __init__(self, client, pose):
        self.client, self.pose = client, pose
        self.goals, self.statuses = {}, {}
        self.moves = 0

    def send_goal(self, goal):
        """Reserve a goal without moving; advance owns the synchronous execution step."""
        uid = f"motion:{len(self.goals)}"
        self.goals[uid] = goal
        self.statuses[uid] = NavigationStatus(NavigationStatusCode.QUEUED, goal_id=uid)
        return uid

    def advance(self, uid):
        """Execute at most one bounded freemap-checked movement and retain its receipt."""
        goal = self.goals[uid]
        if goal.mode == MotionGoalMode.STOP:
            receipt = self.client.command("STOP")
        else:
            if goal.goal_pose is None or goal.goal_pose.frame_id != "map":
                raise ValueError("Isaac motion requires a map-frame goal pose")
            x, y, z, w = goal.goal_pose.orientation_xyzw
            yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            if goal.mode == MotionGoalMode.GO_TO_FRONTIER:
                p, target = self.pose["position"], goal.goal_pose.position
                yaw = math.atan2(target[1] - p[1], target[0] - p[0])
            receipt = self.client.command("move", position=list(goal.goal_pose.position), yaw=yaw,
                                          max_distance=1.7)
            self.moves += 1
        self.pose = receipt["pose"]
        code = {"reached": NavigationStatusCode.REACHED, "running": NavigationStatusCode.RUNNING,
                "blocked": NavigationStatusCode.BLOCKED}[receipt["status"]]
        self.statuses[uid] = NavigationStatus(code, goal_id=uid, current_pose=pose_value(self.pose),
            path_length_remaining=receipt.get("path_length_remaining"), metadata={"receipt": receipt},
            current_velocity=(0.0, 0.0, 0.0))

    def poll_status(self, uid):
        """Read actual execution status without moving the camera."""
        return self.statuses[uid]

    def cancel(self, uid):
        """Confirm quiescence in the synchronous camera backend before cancellation."""
        receipt = self.client.command("STOP")
        self.pose = receipt["pose"]
        self.statuses[uid] = NavigationStatus(NavigationStatusCode.PREEMPTED, goal_id=uid,
            current_pose=pose_value(self.pose), metadata={"receipt": receipt}, current_velocity=(0., 0., 0.))

    def hold(self):
        """Stop camera movement without dispatching another navigation target."""
        self.pose = self.client.command("STOP")["pose"]


class IsaacEvidenceVerifier:
    """Preserve STRIVE constraint prechecks before the existing visual final verifier."""

    def __init__(self, tools, vlm, output):
        self.tools = tools
        self.final = FinalInstructionVerifier(vlm=vlm, evidence_dir=str(output / "verification"))

    def verify(self, *, raw_instruction, instruction_plan, candidate, evidence):
        """Reject missing fresh geometry and leave task progress to the shared reducer."""
        session = self.tools.session
        if evidence["observation_status"] != "current":
            return VerificationResult(False, "need_better_view", task_revision=session.spec.revision,
                target_uid=candidate.uid, view_sufficient_for_stop=False,
                reason="Predicted target surface is not visible in the fresh depth-aligned frame.")
        target = next(t for t in instruction_plan.terminal_targets if t.id == evidence["target_id"])
        checked = session.constraint_evaluator.evaluate_before_final_verifier(
            mapper=session, plan=instruction_plan, target=target, candidate=candidate,
            candidate_obj=session.snapshot.object_by_uid(candidate.uid), evidence=evidence)
        precheck = checked.as_dict()
        precheck.pop("evidence")
        evidence.update(checked.evidence)
        evidence["constraint_precheck"] = precheck
        return self.final.verify(raw_instruction, instruction_plan, candidate, evidence)


def _project_observation_image(packet, observation_map, objects, *, depth=None, object_boxes=None):
    frame, root = packet["frame"], Path(packet["capture_root"])
    if depth is None:
        depth = np.load(root / frame["depth_path"], allow_pickle=False)
    boxes = dict(object_boxes or {})
    for obj in objects:
        if obj.uid not in boxes:
            boxes[obj.uid], _ = visible_projection(observation_map.points(obj.uid), depth,
                frame["intrinsics"], frame["camera_to_map"])
    boxes = {uid: box for uid, box in boxes.items() if box is not None}
    return {"id": str(frame["frame_id"]),
            "image_ref": str((root / frame["rgb_path"]).resolve()),
            "observed_object_uids": list(boxes), "object_boxes": boxes}


class IsaacNavigationTools(NavigationTools):
    """Supply observed instance viewpoints and current candidate-bound visual evidence."""

    def __init__(self, session, motion, observation_map, client, output, vlm):
        self.observation_map, self.client, self.output = observation_map, client, Path(output)
        self.view_generation = 0
        super().__init__(session, motion, None, self.capture_evidence)
        self.verifier = IsaacEvidenceVerifier(self, vlm, self.output)

    def submit_motion(self, task_revision, control_epoch, request_id,
                      candidate_uid=None, viewpoint_uid=None, frontier_uid=None):
        """Reject mismatched versions without motion; retry using the authoritative versions."""
        if task_revision != self.session.spec.revision or control_epoch != self.motion.control_epoch:
            return {"accepted": False, "phase": "rejected", "reason": "proposal_version_mismatch",
                    "task_revision": self.session.spec.revision, "control_epoch": self.motion.control_epoch}
        return super().submit_motion(task_revision, control_epoch, request_id,
                                     candidate_uid, viewpoint_uid, frontier_uid)

    def query_viewpoints(self, candidate_uid):
        """Resolve standoff views from a predicted object, never a ground-truth destination."""
        obj = self.session.snapshot.object_by_uid(candidate_uid)
        if obj is None:
            raise ValueError(f"Unknown observed instance: {candidate_uid}")
        result = self.client.command("viewpoints", position=list(obj.position))
        self.view_generation += 1
        views = tuple(ViewpointSnapshot(
            uid=f"{candidate_uid}:view:{self.view_generation}:{index}",
            pose=pose_value({"position": item["position"], "yaw": item["yaw"]}),
            visible_objects=(candidate_uid,), metadata={"source": item["source"],
                "path_length": item["path_length"], "visibility": "predicted_requires_fresh_verification"})
            for index, item in enumerate(result))
        retained = tuple(v for v in self.session.snapshot.viewpoints if candidate_uid not in v.visible_objects)
        self.session.snapshot = replace(self.session.snapshot, viewpoints=retained + views)
        return super().query_viewpoints(candidate_uid)

    def capture_evidence(self, candidate_uid):
        """Face a predicted object and project its surface into a new actual RGB-D frame."""
        if self.motion.active_request_id is not None:
            raise RuntimeError("Verification requires terminal motion feedback")
        obj = self.session.snapshot.object_by_uid(candidate_uid)
        current = self.motion.controller.pose["position"]
        yaw = math.atan2(obj.position[1] - current[1], obj.position[0] - current[0])
        self.client.command("look", yaw=yaw)
        packet = self.client.command("observe")
        self.motion.controller.pose = packet["pose"]
        frame, root = packet["frame"], Path(packet["capture_root"])
        depth = np.load(root / frame["depth_path"], allow_pickle=False)
        bbox, count = visible_projection(self.observation_map.points(candidate_uid), depth,
            frame["intrinsics"], frame["camera_to_map"])
        evidence = {"observation_status": "current" if bbox is not None else "need_observation",
            "observation_identity": {"candidate_uid": candidate_uid,
                "task_revision": self.session.spec.revision, "observation_id": str(frame["frame_id"])},
            "geometry": {"visible_projected_points": count,
                "projection_failed_in_final_view": bbox is None,
                "distance_to_object": math.dist(obj.position[:2], packet["pose"]["position"][:2]),
                "required_stop_distance": 2.0, "bbox_xyxy": bbox},
            "view_quality_facts": {"visible_projected_points": count},
            "observation_images": [_project_observation_image(packet, self.observation_map,
                self.session.snapshot.objects, depth=depth, object_boxes={candidate_uid: bbox})]}
        # 即使目标投影缺失，旧全景也不能继续冒充当前观察。
        self.session.snapshot = replace(self.session.snapshot, timestamp=time.time(),
            robot_pose=pose_value(packet["pose"]),
            frontiers=tuple(FrontierSnapshot(**item) for item in packet["frontiers"]),
            metadata={**self.session.snapshot.metadata, "panorama_views": 1, "effective_views": 1,
                      "observation_images": evidence["observation_images"]})
        with (self.output / "scene_snapshots.jsonl").open("a") as stream:
            stream.write(json.dumps(asdict(self.session.snapshot), allow_nan=False) + "\n")
        directory = self.output / "verification"
        directory.mkdir(exist_ok=True)
        if bbox is not None:
            image = Image.open(root / frame["rgb_path"]).convert("RGB")
            crop_path = directory / f"{frame['frame_id']:06d}-crop.png"
            overlay_path = directory / f"{frame['frame_id']:06d}-overlay.png"
            image.crop(bbox).save(crop_path)
            ImageDraw.Draw(image).rectangle(bbox, outline="red", width=3)
            image.save(overlay_path)
            evidence.update(current_rgb_with_bbox_path=str(overlay_path), object_crop_path=str(crop_path),
                            object_crop_bbox_xyxy=[0, 0, bbox[2] - bbox[0], bbox[3] - bbox[1]])
        write_json(directory / f"{frame['frame_id']:06d}-evidence.json", evidence)
        return evidence


def refresh_observations(session, observation_map, controller, client, output, panorama_views):
    """Fuse N>=2 panoramic views; internal zero-view refreshes only reproject the map."""
    if (not isinstance(panorama_views, int) or isinstance(panorama_views, bool)
            or panorama_views < 0 or panorama_views == 1):
        raise ValueError("panorama_views must be an integer >= 2, or 0 for internal reprojection")
    start_yaw = controller.pose["yaw"]
    diagnostics = {"observation_mode": "multi", "effective_views": max(1, panorama_views)}
    observation_images = []
    for index in range(panorama_views):
        yaw = start_yaw if index == panorama_views - 1 else start_yaw + 2 * math.pi * (index + 1) / panorama_views
        client.command("look", yaw=yaw)
        packet = client.command("observe")
        report = {**observation_map.add(packet, target_list=session.detector_terms), **diagnostics}
        if index < panorama_views - 1:
            observation_images.append(report["observation_image"])
        with (output / "perception.jsonl").open("a") as stream:
            stream.write(json.dumps(report) + "\n")
        print(json.dumps({"event": "perception", **report}), flush=True)
    if panorama_views == 0:
        client.command("look", yaw=start_yaw)
        packet = client.command("observe")
    controller.pose = packet["pose"]
    objects = observation_map.objects()
    observation_images.append(_project_observation_image(packet, observation_map, objects))
    session.snapshot = SemanticMapSnapshot(time.time(), pose_value(packet["pose"]),
        objects=objects,
        frontiers=tuple(FrontierSnapshot(**item) for item in packet["frontiers"]),
        source="isaac_rgbd_strive_fusion", metadata={"semantic_gt_used": False,
            "frontier_source": "official_freemap_action_proposer", "panorama_views": panorama_views,
            **diagnostics, "observation_images": observation_images})
    with (output / "scene_snapshots.jsonl").open("a") as stream:
        stream.write(json.dumps(asdict(session.snapshot), allow_nan=False) + "\n")


def _poll_motion(runtime, motion):
    """Publish actual owner feedback, retaining request identity after terminal release."""
    request_id = motion.active_request_id
    feedback = motion.poll_active()
    if feedback is not None and runtime is not None:
        runtime.record_event({"event": "motion_feedback", "request_id": request_id,
            "goal_id": feedback.status.goal_id, "candidate_uid": feedback.intent.target_object_uid,
            "status": asdict(feedback.status), "goal": asdict(feedback.motion_goal),
            "intent": asdict(feedback.intent), "tracking_active": feedback.tracking_active,
            "phase": motion.phase, "control_epoch": motion.control_epoch, "held": motion.held,
            "active_request_id": motion.active_request_id,
            "robot_pose": pose_value(motion.controller.pose).as_dict(),
            "moves": motion.controller.moves})
    return feedback


def run_episode(public_episode, output, *, port=9466, vlm="self_hosted", device="cuda:0",
                max_model_calls=60, max_moves=30, panorama_views=4):
    """Run bounded panoramic Agent navigation without evaluator metadata."""
    output = Path(output)
    expected = {"selection_id", "scene_id", "style", "instruction", "start_position", "start_rotation_quat_wxyz"}
    if set(public_episode) != expected:
        raise ValueError("Policy input must contain exactly the public episode fields")
    if min(max_model_calls, max_moves) < 1:
        raise ValueError("Episode budgets must be positive")
    if not isinstance(panorama_views, int) or panorama_views < 2:
        raise ValueError("panorama_views must be an integer >= 2")
    client = IsaacClient(port)
    set_trace_dir(str(output / "model_requests"))
    started = time.monotonic()
    termination = {"status": "running", "policy": "strive_isaac_full_instruction_agent", **public_episode,
        "semantic_authority": "raw_instruction", "structured_interpretation": "revisable_execution_aid",
        "max_model_calls": max_model_calls, "max_moves": max_moves, "panorama_views": panorama_views,
        "observation_mode": "multi", "effective_views": panorama_views,
        "motion_semantics": "freemap_checked_kinematic_camera_not_dynamic_robot",
        "frontier_source": "official_freemap_action_proposer", "model_provider": vlm,
        "model_calls_scope": "initial_interpretation_plus_each_agent_tool_decision; model_usage counts all recorded model calls",
        "moves_scope": "bounded waypoint commands; trajectory records simulator motion increments",
        "semantic_gt_used": False}
    source_root = Path(__file__).resolve().parents[2]
    termination["source_sha256"] = {name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
        for name in ("harness/runtime.py", "harness/tools.py", "task/session.py", "motion/supervisor.py",
                     "backends/isaac/runtime.py", "backends/isaac/perception.py", "cv_utils/sam.py",
                     "mapping/object_fusion.py", "instruction_adapter/verifier.py",
                     "prompting/templates.py", "prompting/schemas.py", "instruction/compiler.py",
                     "instruction_adapter/constraints.py", "instruction_adapter/relation_verifier.py")}
    write_json(output / "policy_result.json", termination)
    motion = runtime = session = controller = None
    initial_compiler = InitialInstructionCompiler(
        max_model_calls=max_model_calls, trace_path=output / "initial_interpretation.jsonl", vlm=vlm)
    try:
        plan = compile_instruction_plan(public_episode["instruction"], available_classes=None,
                                        vlm=vlm, initial_compiler=initial_compiler)
        write_json(output / "instruction_plan.json", plan.as_dict())
        classes = list(dict.fromkeys(list(BACKGROUND_CLASSES) +
            [term for concept in plan.concept_queries for term in concept.detector_terms]))
        write_json(output / "detector_vocabulary.json", {"classes": classes, "source": "generic_context_and_compiled_instruction"})
        # 任务检索词不能固化为基础词表，否则修订后旧词仍会参与检测。
        observation_map = IsaacObservationMap(BACKGROUND_CLASSES, device=device)
        packet = client.command("observe")
        controller = IsaacMotionController(client, packet["pose"])
        session = TaskSession(SemanticMapSnapshot(time.time(), pose_value(packet["pose"])), plan,
                              vlm=vlm, save_dir=str(output), constraint_evaluator=ConstraintEvaluator(
                                  relation_service=DynamicRelationService(vlm=vlm), save_dir=str(output)))
        motion = MotionSupervisor(controller)
        tools = IsaacNavigationTools(session, motion, observation_map, client, output, vlm)
        tools.movement_budget_remaining = max_moves - controller.moves
        runtime = AgentRuntime(tools, output / "tools.jsonl", max_model_calls=max_model_calls, vlm=vlm)
        termination["initial_predicted_target"] = plan.terminal_targets[0].name
        refresh_observations(session, observation_map, controller, client, output, panorama_views)
        while True:
            tools.movement_budget_remaining = max_moves - controller.moves
            # Even the final model call owns its motion through terminal feedback.
            if motion.active_request_id is not None:
                if controller.moves >= max_moves:
                    motion.cancel()
                    _poll_motion(runtime, motion)
                    reason = "movement_budget_exhausted"
                    break
                controller.advance(motion.active_goal_id)
                tools.movement_budget_remaining = max_moves - controller.moves
                feedback = _poll_motion(runtime, motion)
                # Reproject while tracking; map a panorama only after terminal feedback.
                refresh_observations(session, observation_map, controller, client, output,
                                     0 if feedback.tracking_active else panorama_views)
                continue
            if runtime.model_calls >= max_model_calls:
                reason = ("movement_budget_exhausted" if controller.moves >= max_moves
                          else "decision_budget_exhausted")
                break
            # An idle exhausted movement budget still permits verification and finish.
            search_revision = session.search_revision
            outcome = runtime.step()
            result = outcome.get("result", {})
            if (outcome.get("interpretation_changed")
                    or isinstance(result, dict) and result.get("interpretation_changed")
                    or session.search_revision != search_revision):
                refresh_observations(session, observation_map, controller, client, output, panorama_views)
            if outcome["name"] == "propose_finish" and result.get("accepted"):
                reason = "verified_finish"
                break
        receipt = client.command("finish")
        final_packet = client.command("observe")
        completion = tools.propose_finish()
        target_id = completion["verified_target_id"]
        predicted = (next(t for t in session.spec.targets if t.id == target_id)
                     if target_id is not None else session.progress.active_target())
        termination.update(status="completed", reason=reason, task_complete=reason == "verified_finish",
            structured_task_complete=session.progress.completed, completion_review=completion,
            predicted_target=predicted.name, predicted_target_id=predicted.id,
            prediction_task_revision=session.spec.revision,
            prediction_source="current_full_instruction_verification" if target_id else "final_active_interpretation",
            final_task=session.spec.as_dict(), model_calls=runtime.model_calls, moves=controller.moves,
            stop_receipt=receipt, final_frame=final_packet, task_progress=session.progress.as_dict())
    except Exception as error:
        termination.update(status="error", reason=type(error).__name__, error=str(error),
            model_calls=runtime.model_calls if runtime else initial_compiler.model_calls,
            moves=controller.moves if controller else 0,
            task_complete=False, structured_task_complete=session.progress.completed if session else False)
        try:
            if motion and motion.active_request_id is not None:
                motion.cancel()
                _poll_motion(runtime, motion)
            termination["stop_receipt"] = client.command("STOP")
        except Exception as stop_error:
            termination["stop_error"] = str(stop_error)
        raise
    finally:
        termination.update(elapsed_s=time.monotonic() - started, model_usage=counts())
        write_json(output / "policy_result.json", termination)
    return termination
