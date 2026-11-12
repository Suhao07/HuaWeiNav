"""Run guided candidate search against live Isaac observations and motion."""
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
from navigation.guided import GuidedPolicy, GuidedVerification, run_guided_loop
from navigation_core.view_geometry import optical_rotation_towards, rank_view_proposals, score_points_view
from planning.viewpoint_policy import build_view_proposals
from instruction_adapter.view_control import ViewControlState
from harness.runtime import AgentRuntime
from harness.tools import NavigationTools
from instruction.compiler import InitialInstructionCompiler, load_fixed_task
from instruction_adapter.compiler import RUNTIME_CONSTRAINT_TYPES, compile_instruction_plan
from instruction_adapter.task_compat import runtime_plan
from perception.detector_adapter import ground_plan
from instruction_adapter.constraints import ConstraintEvaluator
from instruction_adapter.relation_verifier import DynamicRelationService
from instruction_adapter.verifier import FinalInstructionVerifier
from llm_utils.lvlm_call_tracker import counts, set_trace_dir
from motion.contracts import MotionGoalMode, NavigationIntent, NavigationStatus, NavigationStatusCode
from motion.supervisor import MotionSupervisor
from prior_map.observations import write_json
from scene.contracts import Pose3D, SemanticMapSnapshot, ViewpointSnapshot
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
        self.tools.last_evidence = evidence
        if evidence["observation_status"] != "current":
            return VerificationResult(False, "need_better_view", task_revision=session.spec.revision,
                target_uid=candidate.uid, view_sufficient_for_stop=False,
                reason="Predicted target surface is not visible in the fresh depth-aligned frame.")
        target = next(t for t in instruction_plan.terminal_targets if t.id == evidence["target_id"])
        checked = session.constraint_evaluator.evaluate_before_final_verifier(
            mapper=session, plan=instruction_plan, target=target, candidate=candidate,
            candidate_obj=session.snapshot.object_by_uid(candidate.uid), evidence=evidence)
        if not checked.satisfied:
            return checked.to_verification_result(task_revision=session.spec.revision, target_uid=candidate.uid)
        evidence.update(checked.evidence)
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
            "position": np.asarray(frame["camera_to_map"])[:3, 3].tolist(),
            "rotation": np.asarray(frame["camera_to_map"])[:3, :3].tolist(),
            "observed_object_uids": list(boxes), "object_boxes": boxes}


class IsaacNavigationTools(NavigationTools):
    """Supply observed instance viewpoints and current candidate-bound visual evidence."""

    def __init__(self, session, motion, observation_map, client, output, vlm, exploration_map):
        self.observation_map, self.client, self.output = observation_map, client, Path(output)
        self.exploration_map = exploration_map
        self.vlm = vlm
        self.last_packet = None
        self.last_evidence = {}
        self.deferred_candidates = set()
        self.view_states = {}
        self.view_generation = 0
        self.arrival_tolerance = 0.25
        super().__init__(session, motion, None, self.capture_evidence)
        self.verifier = IsaacEvidenceVerifier(self, vlm, self.output)

    def install_interpretation(self, plan, *, task_revision, search_revision, snapshot, record):
        installed = super().install_interpretation(plan, task_revision=task_revision,
            search_revision=search_revision, snapshot=snapshot, record=record)
        if installed:
            self.deferred_candidates.clear()
        return installed

    def query_scene(self):
        return super().query_scene(deferred_uids=self.deferred_candidates)

    def submit_motion(self, task_revision, control_epoch, request_id,
                      candidate_uid=None, viewpoint_uid=None, frontier_uid=None):
        """Reject mismatched versions without motion; retry using the authoritative versions."""
        if task_revision != self.session.spec.revision or control_epoch != self.motion.control_epoch:
            return {"accepted": False, "phase": "rejected", "reason": "proposal_version_mismatch",
                    "task_revision": self.session.spec.revision, "control_epoch": self.motion.control_epoch}
        return super().submit_motion(task_revision, control_epoch, request_id,
                                     candidate_uid, viewpoint_uid, frontier_uid)

    def _query_paths(self, positions):
        rows = self.client.command("paths", positions=positions)
        if not isinstance(rows, list) or len(rows) != len(positions):
            raise ValueError("paths must return one record per requested position")
        validated = []
        for position, row in zip(positions, rows):
            echoed = np.asarray(row["position"], dtype=float)
            if echoed.shape != (3,) or not np.allclose(echoed, position, rtol=0, atol=1e-6):
                raise ValueError("paths response position does not match its request")
            length = row["path_length"]
            path = np.asarray(row["path"], dtype=float)
            reachable = (length is not None and np.isfinite(length) and float(length) >= 0
                         and path.ndim == 2 and path.shape[1] == 3 and len(path)
                         and np.isfinite(path).all())
            validated.append({**row, "path_length": float(length) if reachable else None})
        return validated

    def query_viewpoints(self, candidate_uid, *, relation_outcomes=()):
        obj = self.session.snapshot.object_by_uid(candidate_uid)
        if obj is None:
            raise ValueError(f"Unknown observed instance: {candidate_uid}")
        verified = {row["predicate_id"] for row in relation_outcomes if row["status"] == "verified"}
        pending = [row for row in relation_outcomes if row["predicate_id"] not in verified
                   and row["status"] == "insufficient_evidence" and row["anchor_uid"]]
        anchors = {row["anchor_uid"]: self.session.snapshot.object_by_uid(row["anchor_uid"])
                   for row in pending}
        frame = self.last_packet["frame"]
        state = self.view_states.setdefault((self.session.spec.revision, candidate_uid), ViewControlState())
        points = self.observation_map.points(candidate_uid)
        observer = np.asarray(self.motion.controller.pose["position"], dtype=float)
        nearest = points[np.argmin(np.linalg.norm(points - observer, axis=1))].copy()
        count = min(8, max(1, int(np.linalg.norm(nearest[:2] - observer[:2]) / 0.1)))
        # Preserve the baseline's surface-first backoff instead of crossing to nearby floor points.
        targets = [nearest]
        if count > 1:
            direction = (observer - nearest) / np.linalg.norm(observer - nearest)
            targets.extend(nearest + direction * (0.1 * i) for i in range(1, count))
        targets = [[float(p[0]), float(p[1]), float(observer[2])] for p in targets]
        for anchor in anchors.values():
            if anchor is not None:
                targets.extend(self.exploration_map.approach_targets(anchor.position))
        if pending:
            targets = [list(position) for position in dict.fromkeys(map(tuple, targets))]
        paths = self._query_paths(targets[:256])
        reachable = [row for row in paths if row["path"] and row["path_length"] is not None]
        proposals = []
        if reachable:
            closest = reachable[0]
            height, width = np.load(Path(self.last_packet["capture_root"]) / frame["depth_path"],
                                    allow_pickle=False).shape
            proposals, _ = build_view_proposals(object_points=points,
                object_position=np.asarray(obj.position), original_bbox=obj.bbox2d_xyxy,
                camera_intrinsic=np.asarray(frame["intrinsics"]), path_points=np.asarray(closest["path"]),
                target_height=self.motion.controller.pose["position"][2], success_distance=2.0,
                stop_criterion=1.0, arrival_tolerance=self.arrival_tolerance,
                image_width=width, image_height=height, pitches=(0.0,),
                stop_reference_position=np.asarray(obj.position))
            for offset in range(0, len(proposals), 256):
                batch = proposals[offset:offset + 256]
                costs = self._query_paths([p["pose"] for p in batch])
                for proposal, row in zip(batch, costs):
                    proposal["travel_distance"] = (row["path_length"] if row["path_length"] is not None
                                                   else math.inf)
        if pending and reachable:
            samples = {tuple(p["pose"]): p["travel_distance"] for p in proposals}
            samples.update({(*row["position"][:2], self.motion.controller.pose["position"][2]):
                            row["path_length"] for row in reachable})
            clouds = {candidate_uid: self.observation_map.points(candidate_uid),
                      **{uid: self.observation_map.points(uid) for uid, anchor in anchors.items()
                         if anchor is not None}}
            proposals = []
            for position, cost in samples.items():
                rotation = optical_rotation_towards(position[:2], obj.position[:2])
                if rotation is None or not math.isfinite(cost):
                    continue
                camera = np.eye(4)
                camera[:3, :3], camera[:3, 3] = rotation, position
                qualities = {}
                for uid, instance in {candidate_uid: obj, **anchors}.items():
                    if instance is None or instance.bbox2d_xyxy is None:
                        continue
                    view = score_points_view(object_points=clouds[uid], original_bbox=instance.bbox2d_xyxy,
                        camera_intrinsic=np.asarray(frame["intrinsics"]), camera_to_map=camera,
                        success_distance=2.0, image_width=width, image_height=height)
                    if view is not None:
                        qualities[uid] = view[0]
                pairs = [{"predicate_id": row["predicate_id"], "anchor_uid": row["anchor_uid"]}
                         for row in pending if row["anchor_uid"] in qualities]
                if candidate_uid not in qualities or not pairs:
                    continue
                quality = qualities[candidate_uid]
                score = min(quality.score, max(qualities[row["anchor_uid"]].score for row in pairs))
                distance = math.dist(position[:2], obj.position[:2])
                proposals.append({**asdict(quality), "pose": list(position), "score": score,
                    "pitch": 0.0, "travel_distance": cost, "covisible_pairs": pairs,
                    "distance_to_target": distance, "visible_surface_distance": quality.distance_to_target,
                    "stop_feasible": distance + self.arrival_tolerance <= 2.0,
                    "reason": "Predicted common view; depth and relation remain unverified."})
        if state.active or pending:
            proposals = [p for p in proposals if p["travel_distance"] > self.arrival_tolerance]
        # 关系补拍优先补齐共同视野，停止距离仍由到达后的独立合同检查。
        proposals = (sorted(proposals, key=lambda p: (-p["score"], p["travel_distance"]))
                     if pending else rank_view_proposals(proposals))
        state.set_proposals(proposals)
        recommended = next((proposal for proposal in state.proposals if not proposal.attempted), None)
        recommended_pose = recommended.pose if recommended else (proposals[0]["pose"] if proposals else None)
        if pending:
            attempted = {tuple(p.pose) for p in state.proposals if p.attempted}
            recommended_pose = next((p["pose"] for p in proposals if tuple(p["pose"]) not in attempted), None)
        self.view_generation += 1
        views = tuple(ViewpointSnapshot(
            uid=f"{candidate_uid}:view:{self.view_generation}:{index}",
            pose=pose_value({"position": item["pose"],
                "yaw": math.atan2(obj.position[1] - item["pose"][1], obj.position[0] - item["pose"][0])}),
            visible_objects=(candidate_uid,), metadata={
                **{key: value for key, value in item.items() if key != "position"},
                "recommended": item["pose"] == recommended_pose,
                "source": "observed_pair_geometry" if pending else "shared_path_sampled_geometry",
                "path_length": item["travel_distance"],
                "distance_source": "observed_instance_position_xy",
                "stop_reference_position": list(obj.position),
                "observation_identity": {"candidate_uid": candidate_uid,
                    "task_revision": self.session.spec.revision, "observation_id": str(frame["frame_id"])},
                "visible_surface_distance_source": "projected_visible_surface_xy",
                "visibility": "predicted_requires_fresh_verification"})
            for index, item in enumerate(proposals))
        retained = tuple(v for v in self.session.snapshot.viewpoints if candidate_uid not in v.visible_objects)
        self.session.snapshot = replace(self.session.snapshot, viewpoints=retained + views)
        return super().query_viewpoints(candidate_uid)

    def capture_evidence(self, candidate_uid):
        """Bind observed-instance XY stop distance to fresh depth-visible evidence.

        Stop distance is ||robot_xy - obj.position_xy||_2, not the nearest cloud
        edge. Fresh visible support remains mandatory; its distance is diagnostic.
        """
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
        self.last_packet = packet
        self.exploration_map.begin_observation()
        self.exploration_map.add(packet)
        self.exploration_map.finish_observation()
        from navigation_core.surface_evidence import measure_instance_surface

        image = Image.open(root / frame["rgb_path"]).convert("RGB")
        masks = self.observation_map.mapper.object_perceiver.sam.segment_current_target(
            np.asarray(image), obj.label)
        _, _, support = visible_projection(self.observation_map.points(candidate_uid), depth,
            frame["intrinsics"], frame["camera_to_map"], return_support=True)
        camera_to_map = np.asarray(frame["camera_to_map"])
        map_to_camera = np.linalg.inv(camera_to_map)
        optical = support @ map_to_camera[:3, :3].T + map_to_camera[:3, 3]
        intrinsic = np.asarray(frame["intrinsics"], dtype=float)
        pixels = optical @ intrinsic.T
        projected = np.column_stack((np.rint(pixels[:, :2] / pixels[:, 2:3]), optical[:, 2]))
        surface_intrinsic = intrinsic.copy()
        # 共享反投影使用 cy-1；光学坐标转换后保持 Isaac 原始像素中心。
        surface_intrinsic[1, 2] += 1
        surface, instance_mask, indices = measure_instance_surface(projected, depth, masks,
            surface_intrinsic, camera_to_map[:3, :3] @ np.diag([1., -1., -1.]),
            camera_to_map[:3, 3], np.asarray(packet["pose"]["position"]),
            depth_tolerance=0.15, min_support_pixels=10)
        support = support[indices]
        count = len(support)
        bbox = None
        if surface["surface_match"]:
            x1, y1, x2, y2 = surface["bbox_xyxy"]
            bbox = [x1, y1, x2 + 1, y2 + 1]
        quality = None if bbox is None else score_points_view(object_points=support, original_bbox=bbox,
            camera_intrinsic=np.asarray(frame["intrinsics"]), camera_to_map=np.asarray(frame["camera_to_map"]),
            success_distance=2.0, image_width=depth.shape[1], image_height=depth.shape[0])
        robot_xy = np.asarray(packet["pose"]["position"][:2])
        distance = (float(np.linalg.norm(np.asarray(obj.position)[:2] - robot_xy))
                    if bbox is not None else None)
        surface_distance = surface["distance_to_object"]
        observation_identity = {"candidate_uid": candidate_uid,
            "task_revision": self.session.spec.revision, "observation_id": str(frame["frame_id"])}
        directory = self.output / "verification"
        directory.mkdir(exist_ok=True)
        surface_path = directory / f"{frame['frame_id']:06d}-surface.npz"
        np.savez_compressed(surface_path, observation_identity=json.dumps(observation_identity),
            rgb=np.asarray(image), depth_m=depth, instance_masks=masks, selected_mask=instance_mask,
            projected_uv_depth=projected, matched_projection_indices=indices,
            camera_intrinsic=surface_intrinsic, camera_to_map=camera_to_map,
            base_position=np.asarray(packet["pose"]["position"]))
        evidence = {"observation_status": "current" if bbox is not None else "need_observation",
            "observation_identity": observation_identity,
            "instance_surface": {**surface, "observation_identity": observation_identity,
                                 "evidence_path": str(surface_path)},
            "geometry": {"visible_projected_points": count,
                "projection_failed_in_final_view": bbox is None,
                "distance_to_object": distance, "distance_source": "observed_instance_position_xy",
                "stop_reference_position": list(obj.position), "observation_identity": observation_identity,
                "visible_surface_distance": surface_distance,
                "visible_surface_distance_source": "depth_visible_surface_xy",
                "required_stop_distance": 2.0, "bbox_xyxy": bbox},
            "view_quality_facts": {**(quality[0].predicted_quality if quality else {}),
                "visible_projected_points": count, "projection_failed": bbox is None},
            "observation_images": [_project_observation_image(packet, self.observation_map,
                self.session.snapshot.objects, depth=depth, object_boxes={candidate_uid: bbox})]}
        state = self.view_states.get((self.session.spec.revision, candidate_uid))
        if state is not None and state.active:
            evidence["view_control"] = state.as_context()
        self.exploration_map.refresh(self.client, packet["pose"])
        fields = self.exploration_map.snapshot_fields(objects=self.session.snapshot.objects)
        self.session.snapshot = replace(self.session.snapshot, timestamp=time.time(),
            robot_pose=pose_value(packet["pose"]), rooms=fields["rooms"], frontiers=fields["frontiers"],
            metadata={**self.session.snapshot.metadata, "observation_images": evidence["observation_images"],
                      "panorama_views": 1, "captured_frames": 1, "perception_frames": 0})
        with (self.output / "scene_snapshots.jsonl").open("a") as stream:
            stream.write(json.dumps(asdict(self.session.snapshot), allow_nan=False) + "\n")
        if bbox is not None:
            import cv2

            crop_path = directory / f"{frame['frame_id']:06d}-crop.png"
            overlay_path = directory / f"{frame['frame_id']:06d}-overlay.png"
            overlay = np.array(image)
            contours, _ = cv2.findContours(instance_mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)
            image = Image.fromarray(overlay)
            image.crop(bbox).save(crop_path)
            ImageDraw.Draw(image).rectangle(bbox, outline="red", width=3)
            image.save(overlay_path)
            evidence.update(current_rgb_with_bbox_path=str(overlay_path), object_crop_path=str(crop_path),
                            object_crop_bbox_xyxy=[0, 0, bbox[2] - bbox[0], bbox[3] - bbox[1]])
        self.last_evidence = evidence
        write_json(directory / f"{frame['frame_id']:06d}-evidence.json", evidence)
        return evidence


def refresh_observations(session, observation_map, controller, client, output, panorama_views, exploration_map,
                         *, covered_headings=None, scan_start_yaw=None):
    if not 0 <= panorama_views <= 12:
        raise ValueError("Panoramic refresh requires zero to twelve adjacent views")
    covered = set() if covered_headings is None else covered_headings
    start_yaw = controller.pose["yaw"] if scan_start_yaw is None else scan_start_yaw
    exploration_map.begin_observation()
    packets, batch, headings, observation_images = [], [], [], []

    def process_batch():
        reports = observation_map.add_panorama(list(batch), session)
        covered.update(headings)
        for packet, report in zip(batch, reports):
            observation_images.append({**report["observation_image"],
                "position": np.asarray(packet["frame"]["camera_to_map"])[:3, 3].tolist(),
                "rotation": np.asarray(packet["frame"]["camera_to_map"])[:3, :3].tolist()})
            with (output / "perception.jsonl").open("a") as stream:
                stream.write(json.dumps(report) + "\n")
            print(json.dumps({"event": "perception", **report}), flush=True)
        batch.clear()
        headings.clear()
        mapper = observation_map.mapper
        return any(mapper._is_target_tag(mapper.objects[index].tag) for index in mapper.current_obj_indices)

    for heading in range(12 if panorama_views else 0):
        yaw = start_yaw - math.pi * heading / 6
        if heading in covered:
            # 三联图投影要求连续视角，已扫描槽位会切断当前批次。
            if batch and process_batch():
                break
            continue
        if abs(math.remainder(yaw - controller.pose["yaw"], 2 * math.pi)) > 1e-6:
            client.command("look", yaw=yaw)
        packet = client.command("observe")
        controller.pose = packet["pose"]
        exploration_map.add(packet)
        packets.append(packet)
        batch.append(packet)
        headings.append(heading)
        if len(batch) == 4 or len(packets) == panorama_views:
            candidate_seen = process_batch()
            if candidate_seen or len(packets) == panorama_views:
                break
    if batch:
        process_batch()
    if not packets:
        packet = client.command("observe")
        controller.pose = packet["pose"]
        exploration_map.add(packet)
    objects = observation_map.objects()
    current_image = _project_observation_image(packet, observation_map, objects)
    if observation_images:
        observation_images[-1] = current_image
    else:
        observation_images.append(current_image)
    mapping_completed = exploration_map.finish_observation()
    exploration_map.refresh(client, packet["pose"])
    fields = exploration_map.snapshot_fields(objects=objects)
    session.snapshot = SemanticMapSnapshot(time.time(), pose_value(packet["pose"]), objects=objects,
        rooms=fields["rooms"], viewpoints=fields["viewpoints"], frontiers=fields["frontiers"],
        source="isaac_rgbd_strive_fusion", metadata={"semantic_gt_used": False,
            "frontier_source": "original_instruct_mapper", "panorama_views": panorama_views,
            "perception_frontend": "original_panoramic_bc", "panorama_complete": len(covered) == 12,
            "observation_schedule": "adaptive4" if panorama_views else "geometry_only",
            "covered_headings": sorted(covered),
            "captured_frames": max(1, len(packets)), "perception_frames": len(packets),
            "mapping_cycle_completed": mapping_completed, "observation_images": observation_images})
    with (output / "scene_snapshots.jsonl").open("a") as stream:
        stream.write(json.dumps(asdict(session.snapshot), allow_nan=False) + "\n")
    return packet


class IsaacGuidedNavigation(GuidedPolicy):
    def __init__(self, tools, runtime, output, *, search_review_moves=10):
        self.tools, self.runtime = tools, runtime
        self.output = Path(output)
        self.selected = None
        self.scan_complete = False
        self.scan_revision = (tools.session.spec.revision, tools.session.search_revision)
        self.need_check_again = False
        self.execution = None
        self.initialize_state(deferred_candidates=tools.deferred_candidates, review_interval=search_review_moves)
        self.candidate_view_states = tools.view_states

    @property
    def step(self):
        return self.tools.motion.controller.moves

    @property
    def revision(self):
        return self.tools.session.spec.revision

    @property
    def arrival_tolerance(self):
        return self.tools.arrival_tolerance

    @property
    def candidate_uid(self):
        return self.selected["candidate_uid"]

    @property
    def view_state(self):
        return self.candidate_view_states[(self.revision, self.candidate_uid)]

    def record(self, event, **fields):
        row = {"event": event, "moves": self.step, "task_revision": self.revision, **fields}
        with (self.output / "decisions.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        self.runtime.record_event(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    def evidence_fingerprint(self, uid):
        from types import SimpleNamespace
        obj = self.tools.session.snapshot.object_by_uid(uid)
        if obj is None or obj.image_ref is None or obj.bbox2d_xyxy is None:
            return None
        with Image.open(obj.image_ref) as image:
            return self.candidate_evidence(SimpleNamespace(rgb=np.asarray(image.convert("RGB")),
                                                          bbox=obj.bbox2d_xyxy))

    def refresh_scene(self):
        self.tools.exploration_map.refresh(self.tools.client, self.tools.motion.controller.pose)
        fields = self.tools.exploration_map.snapshot_fields(objects=self.tools.session.snapshot.objects)
        self.tools.session.snapshot = replace(self.tools.session.snapshot,
                                              rooms=fields["rooms"], frontiers=fields["frontiers"])

    def observation_refreshed(self):
        if self.execution is not None:
            self.execution.mark_completed_scan()

    def select_candidate(self, episode_idx):
        if self.scan_revision != (self.revision, self.tools.session.search_revision):
            self.scan_complete = False
        result = self.tools.query_scene()
        self.record("query_scene", result=result)
        candidates = [c for c in result["candidates"] if c["candidate_uid"] not in self.deferred_candidates]
        self.selected = candidates[0] if candidates else None
        return self.candidate_uid if self.selected else None

    def initial_views(self, uid):
        self.candidate_view_states.setdefault((self.revision, uid), ViewControlState())
        return self.feedback_views(uid)

    def feedback_views(self, uid):
        outcomes = list(self.relation_outcomes.get((self.revision, uid), []))
        for row in self.unresolved_relations(uid):
            if row["status"] != "insufficient_evidence":
                continue
            known = {item["anchor_uid"] for item in outcomes if item["predicate_id"] == row["predicate_id"]}
            for anchor_uid in sorted(self.relation_anchor_uids(uid, row) - known):
                outcomes.append({**row, "anchor_uid": anchor_uid, "evidence_view_ids": [],
                                 "reason": "Observed alternative anchor requires verification."})
        if outcomes:
            self.tools.query_viewpoints(uid, relation_outcomes=outcomes)
        else:
            self.tools.query_viewpoints(uid)
        return [view for view in self.tools.session.snapshot.viewpoints if uid in view.visible_objects]

    def observe_pending(self, episode_idx):
        self.execution.observe(self.execution.panorama_views)

    def node_for_index(self, index):
        return self.tools.exploration_map.nodes[index]

    def exploration_options(self, episode_idx):
        return self.tools.exploration_map.select(choose_room=self.choose_room)

    def select_exploration(self, episode_idx):
        node, source, _ = self.exploration_options(episode_idx)
        return node, source

    def review_options(self, node, source, episode_idx):
        return node, source, list(self.tools.exploration_map.candidates)

    def choose_room(self, rooms):
        from planning.room_selection import choose_semantic_room
        snapshots = self.tools.session.snapshot.rooms
        ids = [room.room_idx for room in rooms]
        payload = {"instruction": self.tools.session.spec.raw_instruction,
                   "robot_pose": self.tools.session.snapshot.robot_pose.as_dict(),
                   "rooms": [asdict(room) for room in snapshots],
                   "objects": [obj.concept_payload() for obj in self.tools.session.snapshot.objects],
                   "candidate_room_ids": ids}
        answer = choose_semantic_room(json.dumps(payload, ensure_ascii=False), ids,
                                      vlm=self.tools.vlm)
        self.record("room_selection", selected_room=answer.final_answer, reason=answer.reason)
        return next(room for room in rooms if room.room_idx == answer.final_answer)

    def submit(self, position, *, candidate_uid=None, node=None, source):
        motion = self.tools.motion
        position = tuple(float(x) for x in position)
        metadata = {"task_revision": self.revision, "decision_source": source}
        if node is not None:
            mode = MotionGoalMode.GO_TO_FRONTIER
            metadata.update(node_index=node.idx, frontier_uid=f"node:{node.idx}")
            yaw = self.tools.motion.controller.pose["yaw"]
        else:
            mode = MotionGoalMode.IMPROVE_VIEW
            obj = self.tools.session.snapshot.object_by_uid(candidate_uid)
            if obj is None:
                raise ValueError("Candidate is no longer observed")
            metadata.update(candidate_evidence_fingerprint=self.evidence_fingerprint(candidate_uid),
                            candidate_action_epoch=self.action_epoch(candidate_uid),
                            target_id=self.selected["target_id"], candidate_role=self.selected["role"])
            yaw = math.atan2(obj.position[1] - position[1], obj.position[0] - position[0])
        self.sequence += 1
        intent = NavigationIntent(mode=mode, target_object_uid=candidate_uid,
            goal_pose=pose_value({"position": position, "yaw": yaw}), metadata=metadata)
        motion.submit(intent, intent.to_motion_goal(), request_id=f"isaac-guided:{self.sequence}",
                      control_epoch=motion.control_epoch)
        if candidate_uid is not None:
            state = self.candidate_view_states[(self.revision, candidate_uid)]
            state.last_selected_index = next((index for index, proposal in enumerate(state.proposals)
                if math.dist(proposal.pose[:2], position[:2]) < 1e-6), None)
        self.record("submitted", candidate_uid=candidate_uid, node_index=metadata.get("node_index"),
                    position=position, decision_source=source)

    def capture_verification(self, scored_position):
        session = self.tools.session
        uid, role, target_id = self.candidate_uid, self.selected["role"], self.selected["target_id"]
        current = self.tools.motion.controller.pose
        pose_gap = {"scored_position": list(scored_position), "execution_position": current["position"],
                    "position_offset_m": math.dist(scored_position[:2], current["position"][:2])}
        if role == "anchor":
            candidates = self.tools.query_scene()["candidates"]
            terminals = [item["candidate_uid"] for item in candidates if item["role"] != "anchor"]
            if terminals:
                status, reason = "terminal_candidate_available", "Observed terminal candidates require verification."
            elif self.scan_complete:
                # 完整语义扫描且候选为空，才构成参照物搜索的负证据。
                status, reason = "searched_no_terminal_found", "Completed panorama; eligible terminal candidate set is empty."
            else:
                status, reason = "search_incomplete", "The panorama has unobserved headings."
            session.anchor_search_ledger.mark(raw_instruction=session.spec.raw_instruction,
                concept_id=target_id, anchor_uid=uid, status=status, step=self.step, reason=reason,
                evidence={"terminal_candidate_uids": terminals,
                          "covered_headings": session.snapshot.metadata["covered_headings"],
                          "panorama_complete": self.scan_complete})
            return GuidedVerification(uid, role, target_id, "anchor_searched", False, pose_gap)
        result = self.tools.verify_target(uid, target_id, self.revision)
        evidence = self.tools.last_evidence
        state = self.view_state
        if state.active:
            state.record_attempt(self.step, evidence, result["decision"],
                semantic_satisfied=result["semantic_satisfied"], reason=result["reason"])
        self.need_check_again = result["decision"] == "need_better_view"
        if self.need_check_again:
            objective = {**result["view_objective"],
                         "hard_stop_constraints": result["hard_constraints"],
                         "required_stop_distance": 2.0,
                         "current_distance_to_object": evidence["geometry"]["distance_to_object"]}
            state.start(uid, objective, evidence)
            state.pin_relation_context(evidence.get("verified_relation_context"))
            if result["semantic_satisfied"]:
                state.pin_visual_evidence(evidence, step=self.step, decision=result["decision"], reason=result["reason"])
        elif result["decision"] == "accept":
            state.reset()
        pose_gap["observation_id"] = evidence["observation_identity"]["observation_id"]
        return GuidedVerification(uid, role, target_id, result["decision"], session.progress.completed, pose_gap,
                                  result["view_objective"].get("relation_outcomes", []))

    def restore_camera(self):
        return True

    def continue_view_control(self):
        return self.need_check_again and not self.view_state.budget_exhausted()


class IsaacGuidedExecution:
    episode_idx = None

    def __init__(self, guided, *, max_moves, panorama_views):
        self.guided = guided
        self.max_moves, self.panorama_views = max_moves, panorama_views
        self.semantic_scan_headings = {}
        self.current_scan = {}
        guided.execution = self

    def observe(self, views):
        tools = self.guided.tools
        position = np.asarray(tools.motion.controller.pose['position']) - tools.observation_map.mapper.initial_position
        key = (tools.session.spec.revision, *np.round(position, 3))
        # 同一位置固定 θ₀，扫描槽位 k 对应 θ₀ − kπ/6，补扫沿用原基准。
        scan = self.semantic_scan_headings.setdefault(key, {
            'start_yaw': tools.motion.controller.pose['yaw'], 'covered': set(),
            'search_revision': tools.session.search_revision})
        if scan['search_revision'] != tools.session.search_revision:
            scan['covered'].clear()
            scan['search_revision'] = tools.session.search_revision
        self.current_scan = scan
        tools.last_packet = refresh_observations(tools.session, tools.observation_map,
            tools.motion.controller, tools.client, tools.output, views, tools.exploration_map,
            covered_headings=scan['covered'], scan_start_yaw=scan['start_yaw'])
        metadata = tools.session.snapshot.metadata
        self.guided.scan_complete = metadata['panorama_complete'] and metadata['mapping_cycle_completed']
        self.guided.scan_revision = (tools.session.spec.revision, tools.session.search_revision)
        if views and not metadata['mapping_cycle_completed']:
            raise RuntimeError('Original mapper scan produced no navigable geometry')

    def mark_completed_scan(self):
        if self.guided.scan_complete and 'node_index' in self.current_scan:
            self.guided.tools.exploration_map.mark_observed(self.current_scan['node_index'])
            del self.current_scan['node_index']

    def initialize(self):
        self.observe(self.panorama_views)
        self.guided.observation_updated()

    def limit_reason(self):
        if self.guided.tools.motion.controller.moves >= self.max_moves:
            return "movement_budget_exhausted"
        if (self.guided.tools.motion.active_request_id is None
                and self.guided.runtime.model_calls >= self.guided.runtime.max_model_calls):
            return "decision_budget_exhausted"
        return None

    def advance(self):
        motion = self.guided.tools.motion
        motion.controller.advance(motion.active_goal_id)

    def after_motion(self, feedback):
        metadata = feedback.intent.metadata
        if feedback.status.succeeded() and (metadata.get('node_index') is not None
                                            or metadata.get('frontier_uid') is not None
                                            or metadata.get('candidate_role') == 'anchor'):
            return
        self.observe(0)
        self.guided.observation_updated()

    def frontier_arrived(self, feedback):
        self.guided.tools.exploration_map.arrive(feedback.intent.metadata['node_index'])
        self.observe(self.panorama_views)
        self.current_scan['node_index'] = feedback.intent.metadata['node_index']

    def frontier_observed(self, feedback):
        self.mark_completed_scan()

    def object_arrived(self, feedback):
        metadata = feedback.intent.metadata
        self.guided.selected = {"candidate_uid": feedback.intent.target_object_uid,
                               "target_id": metadata["target_id"], "role": metadata["candidate_role"]}
        if metadata["candidate_role"] == "anchor":
            self.observe(self.panorama_views)
            self.guided.observation_updated()

    def finish_success(self):
        if not self.guided.tools.propose_finish()["accepted"]:
            raise RuntimeError("Successful STOP requires verified completion and idle motion")


def run_episode(public_episode, output, *, port=9466, vlm="self_hosted", device="cuda:0",
                max_model_calls=60, max_moves=30, panorama_views=12,
                decision_owner="baseline_guided", search_review_moves=10,
                task_spec_path=None, task_spec_sha256=None):
    """Run bounded guided navigation without loading evaluator metadata."""
    output = Path(output)
    expected = {"selection_id", "scene_id", "style", "instruction", "start_position", "start_rotation_quat_wxyz"}
    if set(public_episode) != expected:
        raise ValueError("Policy input must contain exactly the public episode fields")
    if min(max_model_calls, max_moves, panorama_views, search_review_moves) < 1:
        raise ValueError("Episode budgets must be positive")
    if panorama_views > 12:
        raise ValueError("Panoramic refresh supports at most twelve adjacent views")
    if decision_owner != "baseline_guided":
        raise ValueError("Isaac supports only the shared baseline_guided decision owner")
    if (task_spec_path is None) != (task_spec_sha256 is None):
        raise ValueError("Fixed TaskSpec path and SHA256 must be supplied together")
    fixed_task = (load_fixed_task(task_spec_path, task_spec_sha256, public_episode["instruction"],
                                  RUNTIME_CONSTRAINT_TYPES) if task_spec_path is not None else None)
    client = IsaacClient(port)
    set_trace_dir(str(output / "model_requests"))
    started = time.monotonic()
    termination = {"status": "running", "policy": "strive_baseline_guided_isaac", **public_episode,
        "decision_owner": decision_owner, "implementation_tier": "shared_baseline_guided",
        "max_model_calls": max_model_calls, "max_moves": max_moves, "panorama_views": panorama_views,
        "search_review_moves": search_review_moves, "search_review_unit": "bounded_move_commands",
        "motion_semantics": "freemap_checked_kinematic_camera_not_dynamic_robot",
        "frontier_source": "original_instruct_mapper", "model_provider": vlm,
        "mapping_height_band_world_m": [0.08, 1.68],
        "mapping_height_reference": "nominal_world_ground_z0_not_measured_body_support",
        "model_calls_scope": "guided_selection_and_review_only; initial_model_calls reported separately; "
                             "model_usage counts adapter requests and recorded responses by operation; "
                             "SDK-internal transport retries excluded",
        "task_spec_sha256": task_spec_sha256, "interpretation_locked": fixed_task is not None,
        "moves_scope": "bounded waypoint commands; trajectory records simulator motion increments",
        "semantic_gt_used": False}
    source_root = Path(__file__).resolve().parents[2]
    termination["source_sha256"] = {name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
        for name in ("harness/runtime.py", "harness/tools.py", "task/session.py", "motion/supervisor.py",
                     "backends/isaac/runtime.py", "backends/isaac/perception.py", "cv_utils/sam.py",
                     "cv_utils/image_perceiver.py", "cv_utils/stitch.py", "navigation/observation_pipeline.py",
                     "navigation/panoramic_detection.py", "navigation/object_cluster_merger.py",
                     "navigation/object_view_projector.py", "navigation/bbox_refinement.py",
                     "mapping/object_fusion.py", "instruction_adapter/verifier.py", "navigation/guided.py",
                     "backends/isaac/mapper.py", "mapper_with_process_obs.py", "mapping/topology_graph.py",
                     "mapping/room_segmenter.py", "mapping/map_serializer.py", "mapping/frontier_extractor.py",
                     "mapping_utils/geometry.py", "mapping_utils/preprocess.py",
                     "planning/room_policy.py", "planning/viewpoint_policy.py", "planning/exploration_policy.py",
                     "planning/room_selection.py", "navigation_core/view_geometry.py", "instruction/compiler.py",
                     "instruction/contracts.py", "instruction_adapter/task_compat.py", "perception/detector_adapter.py")}
    write_json(output / "policy_result.json", termination)
    motion = runtime = session = controller = initial_compiler = None
    try:
        if fixed_task is None:
            initial_compiler = InitialInstructionCompiler(
                max_model_calls=2, trace_path=output / "initial_interpretation.jsonl", vlm=vlm)
            plan = compile_instruction_plan(public_episode["instruction"], available_classes=None,
                                            vlm=vlm, initial_compiler=initial_compiler)
        else:
            plan = ground_plan(runtime_plan(fixed_task), available_classes=None, vlm=vlm,
                               allow_model_calls=False, use_legacy_similarity_fallback=False)
            plan.diagnostics["initial_model_calls"] = 0
        write_json(output / "instruction_plan.json", plan.as_dict())
        classes = list(dict.fromkeys(list(BACKGROUND_CLASSES) +
            [term for concept in plan.concept_queries for term in concept.detector_terms]))
        write_json(output / "detector_vocabulary.json", {"classes": classes, "source": "generic_context_and_compiled_instruction"})
        # 任务检索词不能固化为基础词表，否则修订后旧词仍会参与检测。
        packet = client.command("observe")
        observation_map = IsaacObservationMap(BACKGROUND_CLASSES, packet, client, save_dir=output,
            device=device, floor_world_z=0.08, ceiling_world_z=1.68)
        controller = IsaacMotionController(client, packet["pose"])
        session = TaskSession(SemanticMapSnapshot(time.time(), pose_value(packet["pose"])), plan,
                              interpretation_locked=fixed_task is not None,
                              vlm=vlm, save_dir=str(output), constraint_evaluator=ConstraintEvaluator(
                                  relation_service=DynamicRelationService(vlm=vlm), save_dir=str(output)))
        motion = MotionSupervisor(controller)
        tools = IsaacNavigationTools(session, motion, observation_map, client, output, vlm, observation_map.mapping)
        runtime = AgentRuntime(tools, output / "tools.jsonl", max_model_calls=max_model_calls, vlm=vlm)
        guided = IsaacGuidedNavigation(tools, runtime, output, search_review_moves=search_review_moves)
        execution = IsaacGuidedExecution(guided, max_moves=max_moves, panorama_views=panorama_views)
        reason = run_guided_loop(guided, execution)
        target_id = guided.selected["target_id"] if guided.selected else None
        target = next((target for target in session.spec.targets if target.id == target_id and target.terminal), None)
        if target is None:
            target = session.progress.active_target()
        candidate_uid = guided.selected["candidate_uid"] if guided.selected else None
        record = (session.verification_ledger.get(session.spec.raw_instruction, candidate_uid)
                  if candidate_uid is not None else None)
        termination.update(predicted_target=record.observed_category if record and record.observed_category else None,
                           requested_target=target.name if target is not None else None,
                           predicted_target_id=target.id if target is not None else None,
                           predicted_candidate_uid=candidate_uid,
                           task_revision=session.spec.revision)
        receipt = client.command("finish")
        final_packet = client.command("observe")
        termination.update(status="completed", reason=reason, task_complete=session.progress.completed,
            model_calls=runtime.model_calls, moves=controller.moves, stop_receipt=receipt,
            final_frame=final_packet, task_progress=session.progress.as_dict())
    except Exception as error:
        termination.update(status="error", reason=type(error).__name__, error=str(error),
            model_calls=runtime.model_calls if runtime else 0,
            moves=controller.moves if controller else 0,
            task_complete=session.progress.completed if session else False)
        try:
            if motion and motion.active_request_id is not None:
                motion.cancel_and_wait()
            termination["stop_receipt"] = client.command("STOP")
        except Exception as stop_error:
            termination["stop_error"] = str(stop_error)
        raise
    finally:
        termination.update(elapsed_s=time.monotonic() - started, model_usage=counts(),
                           initial_model_calls=initial_compiler.model_calls if initial_compiler else 0)
        write_json(output / "policy_result.json", termination)
    return termination
