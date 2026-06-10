import os
import json

import cv2
import habitat
import numpy as np
import open3d as o3d
import quaternion
import torch
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations.maps import \
    colorize_draw_agent_and_fit_to_height
from loguru import logger

from cv_utils.gpt_utils import (ask_gpt_object_in_box,
                                check_again_object_in_bbox,
                                refine_tag_with_target_obj_list)
from cv_utils.stitch import combine_image, image_stitch_and_crop
from cv_utils.visualizer import visualize_mask
from mapper_with_process_obs import Instruct_Mapper
from instruction_adapter.verifier import FinalInstructionVerifier, VerificationResult, candidate_from_object
from instruction_adapter.view_control import ViewControlState
from mapping_utils.geometry import (gpu_cluster_filter, gpu_merge_pointcloud,
                                    gpu_pointcloud_from_array, pointcloud_distance,
                                    project_to_camera)
from mapping_utils.path_planning import path_planning
from mapping_utils.projection import (bresenham_3d, translate_grid_to_point,
                                      translate_point_to_grid,
                                      translate_single_point_to_grid)
from mapping_utils.transform import habitat_rotation
from planning.viewpoint_policy import build_check_again_viewpoints


class HM3D_Objnav_Agent:

    def __init__(self,
                 env: habitat.Env,
                 mapper: Instruct_Mapper,
                 save_dir,
                 do_seg=True,
                 relocate=False,
                 gpt_relocate=True,
                 vlm='cognav'):
        self.env = env
        self.mapper = mapper
        self.episode_samples = 0
        self.planner = ShortestPathFollower(env.sim, 0.15, False)
        self.found_goal = False
        self.save_dir = save_dir
        self.do_seg = do_seg
        self.relocate = relocate
        self.gpt_relocate = gpt_relocate
        self.success_distance = 1.0
        self.stop_criterion = 0.7
        self.vlm = vlm
        self.final_instruction_verifier = FinalInstructionVerifier(vlm=vlm)
        self.view_control_state = ViewControlState()
        self.last_check_again_image_path = ""
        self.last_check_again_evidence = None
        self.final_instruction_accepted_this_step = False
        self.instruction_success = False
        self.instruction_decision = ""
        self.instruction_accept_step = None
        self.accepted_candidate_uid = ""
        self.accepted_relation_edge = {}

    def translate_objnav(self, object_goal):
        if object_goal.lower() == 'plant':
            return "Find the <%s>." % "potted_plant"
        # elif object_goal.lower() == "tv_monitor":
        #     return "Find the <%s>." % "television_set"
        else:
            return "Find the <%s>." % object_goal

    def reset_debug_probes(self):
        self.rgb_trajectory = []
        self.depth_trajectory = []
        self.topdown_trajectory = []
        self.segmentation_trajectory = []

        self.position_trajectory = []
        self.rotation_trajectory = []

        self.gpt_trajectory = []
        self.gptv_trajectory = []
        self.panoramic_trajectory = []

        self.obstacle_affordance_trajectory = []
        self.semantic_affordance_trajectory = []
        self.history_affordance_trajectory = []
        self.action_affordance_trajectory = []
        self.gpt4v_affordance_trajectory = []
        self.affordance_trajectory = []

        self.temporary_pcd = []
        self.temporary_depths = []
        self.angles = []
        self.mapper.current_obj_indices = []

    @property
    def position(self):
        return self.env.sim.get_agent_state().sensor_states['rgb'].position

    @property
    def rotation(self):
        return self.env.sim.get_agent_state().sensor_states['rgb'].rotation

    def reset(self, idx):
        self.episode_samples = idx + 1
        self.episode_steps = 0
        self.obs = self.env.reset()
        self.mapper.initialize(self.position, self.rotation, self.env)
        self.instruct_goal = self.translate_objnav(self.env.current_episode.object_category)
        self.trajectory_summary = ""
        self.reset_debug_probes()

        self.best_distance = 2.0
        self.found_goal = False
        self.need_check_again = False
        self.just_come_back = False

        self.room_final = None
        self.waypoint = None
        self.on_node_flag = False
        self.travel_distance = 0.0
        self.start_end_episode_distance = self.env.get_metrics()['distance_to_goal']

        self.current_node_idx = 0
        self.update_trajectory()
        self.view_control_state.reset()
        self.last_check_again_image_path = ""
        self.last_check_again_evidence = None
        self.final_instruction_accepted_this_step = False
        self.instruction_success = False
        self.instruction_decision = ""
        self.instruction_accept_step = None
        self.accepted_candidate_uid = ""
        self.accepted_relation_edge = {}

    def rotate_panoramic(self, rotate_times=12):
        self.temporary_pcd = []
        temporary_images = []
        temporary_positions, temporary_rotations = [], []
        self.angles = []
        q_identity = quaternion.quaternion(1, 0, 0, 0)
        self.mapper.current_obj_indices = []

        self.B_classes, self.B_boxes, self.B_masks, self.B_confidences, self.B_visualization, \
            self.C_boxes, self.C_masks, self.C_confidences, self.C_visualization = \
                [], [], [], [], [], [], [], [], []

        for i in range(rotate_times):
            if self.env.episode_over:
                logger.info(f'Step: {self.env._elapsed_steps}')
                logger.info(f'Time: {self.env._elapsed_seconds}')
                return

            if self.mapper.current_navigable_pcd is None and self.mapper.current_pcd is None:
                temporary_pcd = gpu_pointcloud_from_array(np.zeros((0, 3)), np.zeros((0, 3)), self.mapper.pcd_device)
            else:
                temporary_pcd = gpu_merge_pointcloud(self.mapper.current_navigable_pcd,
                                                     self.mapper.current_pcd).voxel_down_sample(self.mapper.pcd_resolution)

            self.temporary_pcd.append(temporary_pcd)
            temporary_images.append(self.rgb_trajectory[-1])
            temporary_positions.append(self.mapper.current_position)
            temporary_rotations.append(self.mapper.current_rotation)

            self.angles.append(2 * np.arccos((q_identity.inverse() * self.rotation).w))
            self.obs = self.env.step(3)
            self.update_trajectory()

        temp_depths = self.temporary_depths[-13:-1]
        if self.episode_steps == 13:
            temporary_images[0] = self.rgb_trajectory[-1]
            temporary_positions[0] = self.mapper.current_position
            temporary_rotations[0] = self.mapper.current_rotation
            temp_depths[0] = self.temporary_depths[-1]

        if self.do_seg:
            self.rotate_segmentation(temporary_images, temp_depths,
                                     temporary_positions, temporary_rotations)

        self.just_come_back = False
        logger.info("object indices")
        logger.info(self.mapper.current_obj_indices)

    def rotate_segmentation(self, images, depths, positions, rotations):
        h, w, _ = images[0].shape
        C_objs = []
        os.makedirs(f'{self.save_dir}/episode-{self.episode_samples-1}/detection/step_{self.episode_steps}',
                    exist_ok=True)
        for i in range(12):
            prev = (i - 1) if i > 0 else 11
            nxt = (i + 1) if i < 11 else 0

            comb_img = combine_image(images[prev], images[i], images[nxt],
                                     self.mapper.camera_intrinsic)
            comb_depth = combine_image(depths[prev], depths[i], depths[nxt],
                                       self.mapper.camera_intrinsic)
            depth_vis = np.clip((comb_depth / 5.0 * 255.0), 0, 255).astype(np.uint8)
            B_classes, B_boxes, B_masks, B_confidences, \
                C_classes, C_boxes, C_masks, C_confidences = \
                    self.mapper.object_perceiver.perceive(
                        comb_img,
                        target=self.mapper.target,
                        target_list=getattr(self.mapper, "perception_target_list", None) or self.mapper.target_list,
                        save_dir=self.save_dir,
                        episode_idx=self.episode_samples - 1,
                        episode_step=self.episode_steps,
                    )

            cv2.imwrite(
                f'{self.save_dir}/episode-{self.episode_samples-1}/detection/step_{self.episode_steps}/comb_img_{i}.jpg',
                comb_img)
            cv2.imwrite(
                f'{self.save_dir}/episode-{self.episode_samples-1}/detection/step_{self.episode_steps}/comb_depth_{i}.jpg',
                depth_vis)

            current_pos = [positions[prev], positions[i], positions[nxt]]
            current_rot = [rotations[prev], rotations[i], rotations[nxt]]
            depths_list = [depths[prev], depths[i], depths[nxt]]

            if not (B_boxes is None or B_boxes.shape[0] == 0):

                B_centers = (B_boxes[:, :2] + B_boxes[:, 2:]) * 0.5
                flag = (B_centers[:, 0] >= w) & (B_centers[:, 0] < 2 * w)
                B_classes = B_classes[flag.cpu().numpy()]
                B_centers = B_centers[flag]
                B_boxes = B_boxes[flag]
                B_masks = B_masks[flag]

                if not (B_boxes is None or B_boxes.shape[0] == 0):

                    B_confidences = B_confidences[flag]
                    B_visualization = visualize_mask(comb_img, B_boxes, B_confidences, B_classes, B_masks)

                    cv2.imwrite(
                        f'{self.save_dir}/episode-{self.episode_samples-1}/detection/step_{self.episode_steps}/B_dino_result_{i}.jpg',
                        B_visualization)

                    B_objs = self.mapper.get_object_entities_pano(comb_depth, comb_img,
                        current_pos, current_rot, B_classes, B_boxes, B_masks, B_confidences,
                        depths_list)

                    self.mapper.objects, obj_indices = self.mapper.associate_object_entities(
                        self.mapper.objects, B_objs)
                    self.mapper.current_obj_indices += obj_indices
                    self.mapper.object_pcd = self.mapper.update_object_pcd()

            if C_boxes is None or C_boxes.shape[0] == 0:
                continue

            C_centers = (C_boxes[:, :2] + C_boxes[:, 2:]) * 0.5
            flag = (C_centers[:, 0] >= w) & (C_centers[:, 0] < 2 * w)
            C_classes = C_classes[flag.cpu().numpy()]
            C_centers = C_centers[flag]
            C_boxes = C_boxes[flag]
            C_masks = C_masks[flag]

            if C_boxes is None or C_boxes.shape[0] == 0:
                continue

            C_confidences = C_confidences[flag]
            C_visualization = visualize_mask(comb_img, C_boxes, C_confidences, C_classes, C_masks)

            cv2.imwrite(
                f'{self.save_dir}/episode-{self.episode_samples-1}/detection/step_{self.episode_steps}/C_dino_result_{i}.jpg',
                C_visualization)

            C_objs.append(self.mapper.get_object_entities_pano(comb_depth, comb_img,
                current_pos, current_rot, C_classes, C_boxes, C_masks, C_confidences,
                depths_list))

        # save pc
        obj_pcd = o3d.geometry.PointCloud()
        for i in range(len(C_objs)):
            for obj in C_objs[i]:
                points = obj.pcd_all.point.positions.cpu().numpy()
                colors = obj.pcd_all.point.colors.cpu().numpy()
                new_pcd = o3d.geometry.PointCloud()
                new_pcd.points = o3d.utility.Vector3dVector(points)
                new_pcd.colors = o3d.utility.Vector3dVector(colors)
                o3d.io.write_point_cloud(f'{self.save_dir}/episode-{self.episode_samples-1}/detection/step_{self.episode_steps}/C_dino_pcd_{i}.ply', new_pcd)
                obj_pcd = obj_pcd + new_pcd
        if len(obj_pcd.points) > 0:
            o3d.io.write_point_cloud(f'{self.save_dir}/episode-{self.episode_samples-1}/detection/step_{self.episode_steps}/C_dino_pcd.ply', obj_pcd)
        # original C objs (all)

        # combine the C_objs
        real_C_objs = []
        for i in range(len(C_objs)):
            for obj in C_objs[i]:
                overlap_score = []
                eval_pcd = obj.pcd
                eval_pcd_all = obj.pcd_all
                for prev_obj in real_C_objs:
                    prev_obj_pcd = prev_obj.pcd
                    prev_obj_pcd_all = prev_obj.pcd_all
                    cdist1 = pointcloud_distance(eval_pcd_all, prev_obj_pcd_all)
                    cdist2 = pointcloud_distance(prev_obj_pcd_all, eval_pcd_all)
                    cdist_all = torch.cat([cdist1, cdist2], dim=0)
                    overlap_condition = (cdist_all < 0.1)
                    overlap_condition1 = (cdist1 < 0.1)
                    overlap_condition2 = (cdist2 < 0.1)
                    overlap_score_tmp = (overlap_condition.sum() /
                                          (overlap_condition.shape[0] + 1e-6)).cpu().numpy().item()
                    overlap_score_tmp1 = (overlap_condition1.sum() /
                                            (overlap_condition1.shape[0] + 1e-6)).cpu().numpy().item()
                    overlap_score_tmp2 = (overlap_condition2.sum() /
                                            (overlap_condition2.shape[0] + 1e-6)).cpu().numpy().item()
                    if (overlap_score_tmp1 > 0.85 and overlap_score_tmp2 < 0.85) or (overlap_score_tmp1 < 0.85 and overlap_score_tmp2 > 0.85):
                        overlap_score_tmp = -1.0
                    # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! overlap score: {overlap_score_tmp}, {overlap_score_tmp1}, {overlap_score_tmp2}')
                    overlap_score.append(overlap_score_tmp)
                overlap_flag = [score > 0.25 for score in overlap_score]
                for j in range(len(overlap_flag)):
                    if overlap_flag[j]:
                        eval_pcd = gpu_merge_pointcloud(eval_pcd, real_C_objs[j].pcd)
                        eval_pcd_all = gpu_merge_pointcloud(eval_pcd_all, real_C_objs[j].pcd_all)
                        obj.confidence = max(obj.confidence, real_C_objs[j].confidence)
                obj.pcd = eval_pcd
                obj.pcd_all = eval_pcd_all

                need_to_move = []
                for j in range(len(overlap_flag)):
                    if overlap_flag[j]:
                        need_to_move.append(j)

                real_C_objs = [
                    obj_ for idx, obj_ in enumerate(real_C_objs) if idx not in need_to_move
                ]

                real_C_objs.append(obj)

        # save pc
        obj_pcd = o3d.geometry.PointCloud()
        iii = 0
        for obj in real_C_objs:
            obj.pcd = gpu_cluster_filter(obj.pcd)

            points = obj.pcd.point.positions.cpu().numpy()
            colors = obj.pcd.point.colors.cpu().numpy()
            new_pcd = o3d.geometry.PointCloud()
            new_pcd.points = o3d.utility.Vector3dVector(points)
            new_pcd.colors = o3d.utility.Vector3dVector(colors)
            obj_pcd = obj_pcd + new_pcd

            o3d.io.write_point_cloud(
                f'{self.save_dir}/episode-{self.episode_samples-1}/detection/step_{self.episode_steps}/real_C_objs_{iii}.ply',
                new_pcd)
            iii += 1

        current_pos = self.mapper.current_position[:2]
        nw_ori = self.rotation
        nw_ori = habitat_rotation(nw_ori)
        nw_ori = np.array([-nw_ori[0, 2], -nw_ori[1, 2]])
        for (i, obj) in enumerate(real_C_objs):
            obj.position = np.mean(obj.pcd.point.positions.cpu().numpy(), axis=0)
            center_pos = obj.position[:2]

            # calculate angle
            center_pos = center_pos - current_pos
            nw_ori = nw_ori / np.linalg.norm(nw_ori)
            center_pos = center_pos / np.linalg.norm(center_pos)
            dot_product = np.clip(np.dot(nw_ori, center_pos), -1.0, 1.0)
            angle_rad = np.arccos(dot_product)
            angle = np.degrees(angle_rad)

            cross_product = np.cross(nw_ori, center_pos)
            if cross_product < 0:
                angle = 360 - angle

            angle = angle + 7.5
            if angle >= 360:
                angle -= 360
            image_ind = int(angle / 15)
            final_img = None
            final_box = None
            final_position = self.mapper.current_position
            final_rotation = None
            if image_ind % 2 == 0:
                image_ind = int(image_ind / 2)
                final_img = images[image_ind]
                final_rotation = rotations[image_ind]
            else:
                image_ind = int(image_ind / 2)
                nxt_ind = image_ind + 1
                if nxt_ind == 12:
                    nxt_ind = 0
                final_img = image_stitch_and_crop(images[image_ind],
                                                  images[nxt_ind],
                                                  self.mapper.camera_intrinsic)

                final_rotation = rotations[image_ind]
                deg = np.arccos(final_rotation[0][0])
                if final_rotation[0][2] > 0:
                    deg = np.pi * 2 - deg
                deg += (15 / 180) * np.pi
                final_rotation = np.array([[np.cos(deg), 0, -np.sin(deg)],
                                           [np.sin(deg), 0, np.cos(deg)], [0, 1, 0]])
            cv2.imwrite(
                f"{self.save_dir}/episode-{self.episode_samples-1}/detection/step_{self.episode_steps}/real_C_obj_image_{i}.jpg",
                final_img)

            camera_points = project_to_camera(obj.pcd_all, self.mapper.camera_intrinsic, final_position,
                                              final_rotation)
            camera_points = np.array(camera_points)
            camera_points = camera_points.T
            camera_points = np.array(camera_points[:, :2], dtype=np.int32)
            flag = (camera_points[:, 0] >= 0) & (camera_points[:, 0] < 640) & \
                     (camera_points[:, 1] >= 0) & (camera_points[:, 1] < 480)
            camera_points = camera_points[flag]

            bbox = np.array([np.min(camera_points, axis=0), np.max(camera_points, axis=0)])
            bbox[0] = np.maximum(bbox[0] - 3, 0)
            bbox[1] = np.minimum(bbox[1] + 3, [639, 479])
            final_box = np.array([bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]])
            # save img
            img = final_img.copy()
            cv2.rectangle(img, (bbox[0][0], bbox[0][1]), (bbox[1][0], bbox[1][1]), (0, 255, 0), 2)
            cv2.imwrite(
                f"{self.save_dir}/episode-{self.episode_samples-1}/detection/step_{self.episode_steps}/real_C_obj_image_bbox_{i}.jpg",
                img)

            final_box = torch.tensor(final_box).unsqueeze(0)

            res = ask_gpt_object_in_box(final_img, final_box, self.save_dir, self.episode_samples-1, self.episode_steps, i, self.vlm)

            if res not in self.mapper.object_perceiver.classes:
                res = refine_tag_with_target_obj_list(res, self.mapper.target, self.save_dir, self.episode_samples-1, self.episode_steps, i, self.vlm)

            # 任务目标和检测类别存在同义词差异，例如目标是 tv，
            # GroundingDINO/LLM 可能返回 tv_monitor。这里统一成主目标名，
            # 否则后续 object_found_no_gpt 的目标确认会被精确字符串匹配卡住。
            target_aliases = {self.mapper.target}
            target_aliases.update(getattr(self.mapper, "target_aliases", []) or [])
            if res in target_aliases:
                res = self.mapper.target

            obj.num_list[res] = obj.num_list.pop(obj.tag)
            obj.conf_list[res] = obj.conf_list.pop(obj.tag)
            obj.tag = res

            if res == self.mapper.target:
                obj.confidence = (0.9 * 2 + obj.confidence) / 3
            else:
                obj.confidence = 0.9 / (0.9 + obj.confidence)
            if res == "unknown":
                obj.confidence = 0.0

            if not isinstance(obj.confidence, torch.Tensor):
                obj.confidence = torch.tensor(obj.confidence)
            obj.rgb = final_img
            obj.conf_list[obj.tag] = obj.confidence

            camera_points_real = project_to_camera(obj.pcd, self.mapper.camera_intrinsic, final_position,
                                              final_rotation)
            camera_points_real = np.array(camera_points_real)
            camera_points_real = camera_points_real.T
            camera_points_real = np.array(camera_points_real[:, :2], dtype=np.int32)
            flag = (camera_points_real[:, 0] >= 0) & (camera_points_real[:, 0] < 640) & \
                     (camera_points_real[:, 1] >= 0) & (camera_points_real[:, 1] < 480)
            camera_points_real = camera_points_real[flag]

            if camera_points_real.shape[0] != 0:
                bbox_real = np.array([np.min(camera_points_real, axis=0), np.max(camera_points_real, axis=0)])
                bbox_real = np.array([bbox_real[0][0], bbox_real[0][1], bbox_real[1][0], bbox_real[1][1]])
                obj.bbox = bbox_real

        # move all self.mapper.target_objects to the end
        real_C_objs_sorted = []
        target_objs = []
        for obj in real_C_objs:
            if obj.tag == self.mapper.target:
                target_objs.append(obj)
            else:
                real_C_objs_sorted.append(obj)
        real_C_objs_sorted += target_objs
        real_C_objs = real_C_objs_sorted

        self.mapper.objects, obj_indices = self.mapper.associate_object_entities(
            self.mapper.objects, real_C_objs)
        self.mapper.current_obj_indices += obj_indices
        self.object_pcd = self.mapper.update_object_pcd()

        self.mapper.current_obj_indices = list(set(self.mapper.current_obj_indices))

    def concat_panoramic(self, images):
        try:
            height, width = images[0].shape[0], images[0].shape[1]
        except:
            height, width = 480, 640
        background_image = np.zeros((2 * height + 3 * 10, 3 * width + 4 * 10, 3), np.uint8)
        copy_images = np.array(images, dtype=np.uint8)
        for i in range(len(copy_images)):
            if i % 2 != 0:
                row = (i // 6)
                col = ((i % 6) // 2)
                copy_images[i] = cv2.putText(copy_images[i], "Direction %d" % i, (100, 100),
                                             cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 6,
                                             cv2.LINE_AA)
                background_image[10 * (row + 1) + row * height:10 * (row + 1) + row * height +
                                 height:, col * width + col * 10:col * width + col * 10 +
                                 width, :] = copy_images[i]

        return background_image

    def update_trajectory(self, on_node_flag=False):
        self.episode_steps += 1
        self.metrics = self.env.get_metrics()
        self.rgb_trajectory.append(cv2.cvtColor(self.obs['rgb'], cv2.COLOR_BGR2RGB))
        self.depth_trajectory.append((self.obs['depth'] / 5.0 * 255.0).astype(np.uint8))
        self.temporary_depths.append(self.obs['depth'].copy())

        # 不同 Habitat 版本的 top_down_map 可能不存在；缺失时用空白图保持可视化链路可写。
        topdown_metric = self.metrics.get('top_down_map')
        if topdown_metric is None:
            topdown_image = np.full((1024, 1024, 3), 255, dtype=np.uint8)
        else:
            topdown_image = cv2.cvtColor(
                colorize_draw_agent_and_fit_to_height(topdown_metric, 1024),
                cv2.COLOR_BGR2RGB)
            topdown_image = cv2.flip(topdown_image, 0)
        # Habitat 新旧版本对 SoftSPL 的 key 命名不同。
        soft_spl = self.metrics.get('soft_spl', self.metrics.get('softspl', 0.0))
        text = f"Success:{self.metrics['success']:.2f}, SPL:{self.metrics['spl']:.2f}, SoftSPL:{soft_spl:.2f}, DTS:{self.metrics['distance_to_goal']:.2f}, Step:{self.episode_steps}, Goal:{self.env.current_episode.object_category}"
        topdown_image = cv2.putText(
            topdown_image, text, (0, 100),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        self.topdown_trajectory.append(topdown_image)

        self.position_trajectory.append(self.position)
        self.rotation_trajectory.append(self.rotation)

        if len(self.position_trajectory) > 1:
            pos1 = self.position_trajectory[-1]
            pos2 = self.position_trajectory[-2]
            pos1 = [pos1[0], pos1[2]]
            pos2 = [pos2[0], pos2[2]]
            self.travel_distance += np.linalg.norm((np.array(pos1) - np.array(pos2)))

        self.mapper.update(self.rgb_trajectory[-1], self.obs['depth'], self.position, self.rotation, self.episode_samples - 1, self.episode_steps, on_node_flag, self.current_node_idx)

        os.makedirs(f'{self.save_dir}/episode-{self.episode_samples-1}/rgb', exist_ok=True)
        os.makedirs(f'{self.save_dir}/episode-{self.episode_samples-1}/depth', exist_ok=True)
        os.makedirs(f'{self.save_dir}/episode-{self.episode_samples-1}/topdown', exist_ok=True)
        cv2.imwrite(
            f"{self.save_dir}/episode-{self.episode_samples-1}/rgb/monitor-rgb_{self.episode_steps}.jpg",
            self.rgb_trajectory[-1])
        cv2.imwrite(
            f"{self.save_dir}/episode-{self.episode_samples-1}/depth/monitor-depth_{self.episode_steps}.jpg",
            self.depth_trajectory[-1])
        cv2.imwrite(
            f"{self.save_dir}/episode-{self.episode_samples-1}/topdown/monitor-topdown_{self.episode_steps}.jpg",
            self.topdown_trajectory[-1])
        # cv2.imwrite("monitor-rgb.jpg", self.rgb_trajectory[-1])

        if self.episode_steps == 499:
            self.obs = self.env.step(0)
            self.update_trajectory()
            logger.info('Episode over!!!!!')

    def save_trajectory(self, dir="./tmp_objnav/"):
        import imageio
        os.makedirs(dir, exist_ok=True)

        def fit_frame(frame, target_shape):
            # top-down map 会随地图宽度变化，同一个 mp4 writer 必须接收固定尺寸 frame。
            target_h, target_w = target_shape[:2]
            if frame.shape[:2] == (target_h, target_w):
                return frame
            return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

        self.mapper.save_pointcloud_debug(dir)
        fps_writer = imageio.get_writer(dir + "fps.mp4", fps=4)
        dps_writer = imageio.get_writer(dir + "depth.mp4", fps=4)
        metric_writer = imageio.get_writer(dir + "metrics.mp4", fps=4)
        rgb_shape = self.rgb_trajectory[0].shape
        depth_shape = self.depth_trajectory[0].shape
        metric_shape = self.topdown_trajectory[0].shape
        for i, img, dep, met in zip(np.arange(len(self.rgb_trajectory)), self.rgb_trajectory,
                                    self.depth_trajectory, self.topdown_trajectory):
            fps_writer.append_data(cv2.cvtColor(fit_frame(img, rgb_shape), cv2.COLOR_BGR2RGB))
            dps_writer.append_data(fit_frame(dep, depth_shape))
            metric_writer.append_data(cv2.cvtColor(fit_frame(met, metric_shape), cv2.COLOR_BGR2RGB))

        fps_writer.close()
        dps_writer.close()
        # seg_writer.close()
        metric_writer.close()

        # rm the top-down folder
        folder_path = f'{self.save_dir}/episode-{self.episode_samples-1}/topdown'
        os.system(f'rm -r {folder_path}')

    def make_plan(self, rotate=True, failed=False):
        if rotate == True:
            self.rotate_panoramic()
        self.chainon_answer = self.query_chainon()
        self.gpt4v_answer = self.query_gpt4v()
        self.gpt4v_pcd = o3d.t.geometry.PointCloud(self.mapper.pcd_device)
        self.gpt4v_pcd = gpu_merge_pointcloud(self.gpt4v_pcd, self.temporary_pcd[self.gpt4v_answer])
        self.found_goal = bool(self.chainon_answer['Flag'])
        self.affordance_pcd, self.colored_affordance_pcd = self.mapper.get_objnav_affordance_map(
            self.chainon_answer['Action'],
            self.chainon_answer['Landmark'],
            self.gpt4v_pcd,
            self.chainon_answer['Flag'],
            failure_mode=failed)
        self.semantic_afford, self.history_afford, self.action_afford, self.gpt4v_afford, self.obs_afford = self.mapper.get_debug_affordance_map(
            self.chainon_answer['Action'], self.chainon_answer['Landmark'], self.gpt4v_pcd)
        if self.affordance_pcd.max() == 0:
            self.affordance_pcd, self.colored_affordance_pcd = self.mapper.get_objnav_affordance_map(
                self.chainon_answer['Action'],
                self.chainon_answer['Landmark'],
                self.gpt4v_pcd,
                False,
                failure_mode=failed)
            self.found_goal = False

        self.affordance_map, self.colored_affordance_map = project_costmap(
            self.mapper.navigable_pcd, self.affordance_pcd, self.mapper.grid_resolution)
        self.target_point = self.mapper.navigable_pcd.point.positions[
            self.affordance_pcd.argmax()].cpu().numpy()
        self.plan_position = self.mapper.current_position.copy()
        target_index = translate_point_to_grid(self.mapper.navigable_pcd, self.target_point,
                                               self.mapper.grid_resolution)
        start_index = translate_point_to_grid(self.mapper.navigable_pcd,
                                              self.mapper.current_position,
                                              self.mapper.grid_resolution)
        self.path = path_planning(self.affordance_map, start_index, target_index)
        self.path = [
            translate_grid_to_point(self.mapper.navigable_pcd,
                                    np.array([[waypoint.y, waypoint.x, 0]]),
                                    self.mapper.grid_resolution)[0] for waypoint in self.path
        ]
        if len(self.path) == 0:
            self.waypoint = self.mapper.navigable_pcd.point.positions.cpu().numpy()[np.argmax(
                self.affordance_pcd)]
            self.waypoint[2] = self.mapper.current_position[2]
        elif len(self.path) < 5:
            self.waypoint = self.path[-1]
            self.waypoint[2] = self.mapper.current_position[2]
        else:
            self.waypoint = self.path[4]
            self.waypoint[2] = self.mapper.current_position[2]

        self.affordance_trajectory.append(self.colored_affordance_pcd)
        self.obstacle_affordance_trajectory.append(self.obs_afford)
        self.semantic_affordance_trajectory.append(self.semantic_afford)
        self.history_affordance_trajectory.append(self.history_afford)
        self.action_affordance_trajectory.append(self.action_afford)
        self.gpt4v_affordance_trajectory.append(self.gpt4v_afford)

    def step(self):
        to_target_distance = np.sqrt(np.sum(np.square(self.mapper.current_position -
                                                      self.waypoint)))
        if to_target_distance < 0.6 and len(self.path) > 0:
            self.path = self.path[min(5, len(self.path) - 1):]
            if len(self.path) < 3:
                self.waypoint = self.path[-1]
                self.waypoint[2] = self.mapper.current_position[2]
            else:
                self.waypoint = self.path[2]
                self.waypoint[2] = self.mapper.current_position[2]

        pid_waypoint = self.waypoint + self.mapper.initial_position
        pid_waypoint = np.array(
            [pid_waypoint[0],
             self.env.sim.get_agent_state().position[1], pid_waypoint[1]])

        # act = self.planner.get_next_action(pid_waypoint)

        move_distance = np.sqrt(np.sum(np.square(self.mapper.current_position -
                                                 self.plan_position)))
        if (act == 0 or move_distance > 3.0) and not self.found_goal:
            self.make_plan(rotate=True)
            pid_waypoint = self.waypoint + self.mapper.initial_position
            pid_waypoint = np.array(
                [pid_waypoint[0],
                 self.env.sim.get_agent_state().position[1], pid_waypoint[1]])
            act = self.planner.get_next_action(pid_waypoint)
        if act == 0 and not self.found_goal:
            self.make_plan(False, True)
            pid_waypoint = self.waypoint + self.mapper.initial_position
            pid_waypoint = np.array(
                [pid_waypoint[0],
                 self.env.sim.get_agent_state().position[1], pid_waypoint[1]])
            act = self.planner.get_next_action(pid_waypoint)
            logger.info("Warning: Failure locomotion and action = %d" % act)
        if not self.env.episode_over:
            self.obs = self.env.step(act)
            self.update_trajectory()

    def _merge_temporary_pointclouds(self):
        merged_pcd = o3d.t.geometry.PointCloud(self.mapper.pcd_device)
        for pcd in self.temporary_pcd:
            merged_pcd = gpu_merge_pointcloud(merged_pcd, pcd)
        return merged_pcd

    def _save_obs_pointcloud(self, pcd, idx, step, path_idx=None):
        save_pcd = o3d.geometry.PointCloud()
        save_pcd.points = o3d.utility.Vector3dVector(pcd.point.positions.cpu().numpy())
        save_pcd.colors = o3d.utility.Vector3dVector(pcd.point.colors.cpu().numpy())
        os.makedirs(f'{self.save_dir}/episode-{idx}/obs', exist_ok=True)
        if path_idx is None:
            file_path = f'{self.save_dir}/episode-{idx}/obs/obs_{step}.ply'
        else:
            file_path = f'{self.save_dir}/episode-{idx}/obs/obs_{step}_{path_idx}.ply'
        o3d.io.write_point_cloud(file_path, save_pcd)

    def _log_mapper_state_before_after_get_nodes(self, step, node, idx):
        logger.info("\n \n --------------------------------------------------")
        logger.info(f'Current step: {step}')
        logger.info("before {}", self.mapper.node_cnt)
        self.mapper.get_nodes(self.temporary_pcd, self.angles, node, episode_idx=idx, step=step)
        logger.info("after {}", self.mapper.node_cnt)
        logger.info(self.mapper.get_nodes_states())
        logger.info(self.mapper.get_nodes_positions())
        logger.info("current position: {}", self.mapper.current_position)

    def make_plan_mod_no_relocate(self,
                                  rotate=True,
                                  failed=False,
                                  initial=False,
                                  node=None,
                                  idx=None,
                                  use_gpt_relocate=None):
        self.on_node_flag = True
        if use_gpt_relocate is None:
            use_gpt_relocate = self.gpt_relocate

        self.rotate_panoramic()
        if self.env.episode_over:
            return False, False

        self.current_pcd = self._merge_temporary_pointclouds()
        step = self.episode_steps
        self._save_obs_pointcloud(self.current_pcd, idx=idx, step=step)

        self._log_mapper_state_before_after_get_nodes(step, node, idx)
        self.mapper.update_obj(self.current_node_idx, self.mapper.current_obj_indices)

        logger.info("-------------------Check Whether The Object is Found-------------------")
        self.found_goal, self.object_final = self.mapper.object_found_no_gpt(self.instruct_goal,
                                                                             idx=idx,
                                                                             step=step)
        if self.found_goal:
            self.found_goal_position = self.mapper.current_position
            self.find_final_waypoint()
            self.whether_to_check_again()
            self.path[:, 2] = self.mapper.current_position[2]
            self.waypoint_final = None
            self.found_goal = True

            return True, self.found_goal

        current_node = self.mapper.nodes[self.current_node_idx]
        room_node = self.mapper.room_nodes[current_node.room_idx]
        self.waypoint = self.mapper.explore_in_room(room_node)
        if self.waypoint is not None:
            self.current_node_idx = self.waypoint.idx
            self.path = np.array([self.waypoint.position])
            self.path_index = 0
            self.waypoint_final = self.waypoint
            logger.info(f'Final waypoint: {self.waypoint_final.position}')

            self.mapper.traj.append(self.waypoint.idx)
            self.mapper.change_state(self.waypoint)

            self.waypoint = self.path[0]


            return True, self.found_goal

        if self.waypoint is None:
            logger.info("----------------Relocate After Fully Explored----------------")
            room_state = [room_node.state for room_node in self.mapper.room_nodes]
            if 0 not in room_state:
                logger.info("Fully Explored, Visit unvisited nodes!")
                self.waypoint = self.mapper.explore_after_fully_explored()
            elif use_gpt_relocate:
                self.room_final = self.mapper.get_candidate_room_fully_explored(self.instruct_goal,
                                                                                idx=idx,
                                                                                step=step)
                self.waypoint = self.mapper.find_closet_viewpoint_in_room(self.room_final)
            else:
                self.room_final = self.mapper.get_candidate_room_fully_explored_by_distance(self.instruct_goal,
                                                                                            idx=idx,
                                                                                            step=step)
                self.waypoint = self.mapper.find_closet_viewpoint_in_room(self.room_final)

            if self.waypoint is None:
                logger.info("No unvisited nodes, Fully Explored!!!!!")
                return False, self.found_goal

            self.current_node_idx = self.waypoint.idx
            self.waypoint_final = self.waypoint
            logger.info(f'Final waypoint: {self.waypoint_final.position}')

            self.mapper.traj.append(self.waypoint.idx)
            self.mapper.change_state(self.waypoint)

            self.path, self.path_node_idx = self.mapper.get_path(self.waypoint)

            self.path[:, 2] = self.mapper.current_position[2]
            if len(self.path) == 1:
                self.waypoint = self.path[0]
                self.path_node_idx = self.path_node_idx[0]
                self.path_index = 0
            else:
                self.path = self.path[1:]
                self.waypoint = self.path[0]
                self.path_index = 0

            logger.info(f'Path: {self.path}')

            return True, self.found_goal

    def make_plan_mod_relocate(self,
                                  rotate=True,
                                  failed=False,
                                  initial=False,
                                  node=None,
                                  idx=None):
        self.on_node_flag = True

        self.rotate_panoramic()
        if self.env.episode_over:
            return False, False

        self.current_pcd = self._merge_temporary_pointclouds()
        step = self.episode_steps
        self._save_obs_pointcloud(self.current_pcd, idx=idx, step=step)

        self._log_mapper_state_before_after_get_nodes(step, node, idx)
        self.mapper.update_obj(self.current_node_idx, self.mapper.current_obj_indices)

        logger.info("-------------------Check Whether The Object is Found-------------------")
        # self.found_goal, self.object_final = self.mapper.object_found(self.instruct_goal, idx=idx, step=step)
        self.found_goal, self.object_final = self.mapper.object_found_no_gpt(self.instruct_goal,
                                                                             idx=idx,
                                                                             step=step)
        if self.found_goal:
            self.found_goal_position = self.mapper.current_position
            self.find_final_waypoint()
            self.whether_to_check_again()

            # self.waypoint = self.object_final.find_closest(self.found_goal_position)
            # self.waypoint[2] = self.found_goal_position[2]
            # self.path = np.array([self.waypoint])
            # self.path_index = 0

            self.path[:, 2] = self.mapper.current_position[2]
            self.waypoint_final = None
            self.found_goal = True

            return True, self.found_goal

        current_node = self.mapper.nodes[self.current_node_idx]
        room_node = self.mapper.room_nodes[current_node.room_idx]

        # # if enter a new room, decide whether to go back
        if self.waypoint is not None:
            previous_node = self.mapper.nodes[self.current_node_idx]
            if previous_node.room_idx != current_node.room_idx:
                logger.info('Accidentally enter a new room!!!')

        self.waypoint = self.mapper.explore_in_room_relocate(room_node)
        if self.waypoint is not None:
            self.current_node_idx = self.waypoint.idx
            self.path = np.array([self.waypoint.position])
            self.path_index = 0
            self.waypoint_final = self.waypoint
            logger.info(f'Final waypoint: {self.waypoint_final.position}')

            self.mapper.traj.append(self.waypoint.idx)
            self.mapper.change_state(self.waypoint)

            self.waypoint = self.path[0]

            return True, self.found_goal

        if self.waypoint is None:
            logger.info("----------------Relocate----------------")
            # ----------------relocate----------------
            # self.waypoint_final, self.object_final, found_goal = self.mapper.get_candidate_node(self.instruct_goal, idx=idx, step=step)
            room_state = [room_node.state for room_node in self.mapper.room_nodes]
            if 0 not in room_state:
                logger.info("Fully Explored, Visit unvisited nodes!")
                self.waypoint = self.mapper.explore_after_fully_explored()
            else:
                self.room_final = self.mapper.get_candidate_room_fully_explored(self.instruct_goal,
                                                                                idx=idx,
                                                                                step=step)

                self.waypoint = self.mapper.find_closet_viewpoint_in_room(self.room_final)

            if self.waypoint is None:
                logger.info("No unvisited nodes, Fully Explored!!!!!")
                return False, self.found_goal

            self.current_node_idx = self.waypoint.idx
            self.waypoint_final = self.waypoint
            logger.info(f'Final waypoint: {self.waypoint_final.position}')
            # if self.room_final is None:
            #     return False, self.found_goal

            self.mapper.traj.append(self.waypoint.idx)
            self.mapper.change_state(self.waypoint)

            self.path, self.path_node_idx = self.mapper.get_path(self.waypoint)

            self.path[:, 2] = self.mapper.current_position[2]
            if len(self.path) == 1:
                self.waypoint = self.path[0]
                self.path_node_idx = self.path_node_idx[0]
                self.path_index = 0
            else:
                self.path = self.path[1:]
                self.waypoint = self.path[0]
                self.path_index = 0

            logger.info(f'Path: {self.path}')

            return True, self.found_goal

    def make_plan_mod_process(self,
                              rotate=True,
                              failed=False,
                              initial=False,
                              node=None,
                              idx=None,
                              path_idx=None):
        self.on_node_flag = True

        if self.mapper.process_obs_pcd.is_empty():
            return

        step = self.episode_steps
        self._save_obs_pointcloud(self.mapper.process_obs_pcd, idx=idx, step=step, path_idx=path_idx)

        self.mapper.get_nodes_process(node, idx=idx, step=step, path_idx=path_idx)

    def _project_object_bbox_on_current_view(self):
        current_depth = self.obs['depth'].copy()
        camera_points = project_to_camera(self.object_final.pcd, self.mapper.camera_intrinsic,
                                          self.mapper.current_position,
                                          self.mapper.current_rotation)
        camera_points = np.array(camera_points)
        camera_points = camera_points.T
        depth = np.array(camera_points[:, 2], dtype=np.float32)
        camera_points = np.array(camera_points[:, :2], dtype=np.int32)
        flag = (camera_points[:, 0] >= 0) & (camera_points[:, 0] < 640) & \
               (camera_points[:, 1] >= 0) & (camera_points[:, 1] < 480)
        camera_points = camera_points[flag]
        depth = depth[flag]

        current_depth = current_depth[camera_points[:, 1], camera_points[:, 0]][:, 0]
        current_depth = np.array(current_depth, dtype=np.float32)
        depth_flag = (depth - current_depth) < 0.2
        camera_points = camera_points[depth_flag]

        if len(camera_points) == 0:
            return None, {}

        bbox = np.array([np.min(camera_points, axis=0), np.max(camera_points, axis=0)])
        # x1 y1 x2 y2
        bbox_np = np.array([bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]], dtype=np.float32)
        width = max(1.0, float(bbox_np[2] - bbox_np[0]))
        height = max(1.0, float(bbox_np[3] - bbox_np[1]))
        geometry = {
            "bbox_xyxy": [float(x) for x in bbox_np.tolist()],
            "bbox_center_norm": [
                float(((bbox_np[0] + bbox_np[2]) / 2.0) / 640.0),
                float(((bbox_np[1] + bbox_np[3]) / 2.0) / 480.0),
            ],
            "bbox_area_ratio": float((width * height) / (640.0 * 480.0)),
            "visible_projected_points": int(len(camera_points)),
        }
        return torch.tensor(bbox_np).unsqueeze(0), geometry

    @staticmethod
    def _view_quality_facts(geometry):
        geometry = dict(geometry or {})
        center = geometry.get("bbox_center_norm") or []
        area = geometry.get("bbox_area_ratio")
        cx = cy = None
        center_offset_norm = None
        border_margin_norm = None
        if isinstance(center, (list, tuple)) and len(center) >= 2:
            try:
                cx = float(center[0])
                cy = float(center[1])
                center_offset_norm = float(np.linalg.norm(np.array([cx - 0.5, cy - 0.5], dtype=float)))
                border_margin_norm = float(min(cx, 1.0 - cx, cy, 1.0 - cy))
            except Exception:
                cx = cy = None
                center_offset_norm = None
                border_margin_norm = None
        target_position_hint = "unknown"
        if cx is not None and cy is not None:
            horizontal = "center" if abs(cx - 0.5) < 0.15 else ("left" if cx < 0.5 else "right")
            vertical = "middle" if 0.25 <= cy <= 0.75 else ("upper" if cy < 0.25 else "lower")
            target_position_hint = f"{vertical}-{horizontal}"
        return {
            "bbox_center_norm": center,
            "bbox_area_ratio": area,
            "visible_projected_points": geometry.get("visible_projected_points"),
            "distance_to_object": geometry.get("distance_to_object"),
            "center_offset_norm": center_offset_norm,
            "border_margin_norm": border_margin_norm,
            "target_position_hint": target_position_hint,
            "projection_failed": bool(geometry.get("projection_failed_in_final_view", False)),
            "instruction": (
                "Use these generic camera-quality facts to judge final stop evidence. "
                "Large center_offset, small border_margin, tiny bbox area, or failed projection "
                "means the candidate may need a clearer view even when semantics are correct."
            ),
        }

    def _estimate_distance_to_object_final(self):
        try:
            positions = self.object_final.pcd.point.positions.cpu().numpy()
            if len(positions) == 0:
                return None
            return float(np.min(np.linalg.norm(positions[:, :2] - self.mapper.current_position[:2], axis=1)))
        except Exception:
            return None

    def check_again(self, episode_step):
        bbox, geometry = self._project_object_bbox_on_current_view()
        if bbox is None:
            logger.info(f"Abort check again due to visibility.")
            self.last_check_again_evidence = None
            return True

        img = self.rgb_trajectory[-1].copy()

        img_vis = visualize_mask(img, bbox)
        os.makedirs(f'{self.save_dir}/episode-{self.episode_samples-1}/check_again', exist_ok=True)
        self.last_check_again_image_path = (
            f'{self.save_dir}/episode-{self.episode_samples-1}/check_again/rgb_{episode_step}.jpg'
        )
        cv2.imwrite(
            self.last_check_again_image_path,
            img_vis)
        instruction_mode = (
            getattr(self.mapper, "instruction_plan", None) is not None
            or getattr(self.mapper, "instruction_spec", None) is not None
        )
        if instruction_mode:
            # 中文说明：指令模式下不再用独立的类别复核作为语义结论。
            # check_again 只采集更清晰证据；candidate 是否是目标、关系是否
            # 成立、当前视角是否足够 stop，统一交给 final verifier 判断。
            flag = True
            answer_path = f'{self.save_dir}/episode-{self.episode_samples - 1}/check_again/answer_{episode_step}.txt'
            with open(answer_path, "w", encoding="utf-8") as f:
                f.write("Evidence-only check_again for instruction mode. Unified final verifier owns semantic decision.\n")
        else:
            flag = check_again_object_in_bbox(
                img_vis=img_vis,
                target=self.mapper.target,
                save_dir=self.save_dir,
                episode_idx=self.episode_samples - 1,
                episode_step=episode_step,
                vlm=self.vlm,
            )
        candidate = candidate_from_object(
            self.object_final,
            canonical_label=getattr(self.mapper, "target", ""),
            step=episode_step,
        )
        distance = self._estimate_distance_to_object_final()
        geometry_for_evidence = {
            **dict(geometry or {}),
            "source": "check_again",
        }
        if distance is not None:
            geometry_for_evidence["distance_to_object"] = distance
        # check_again 的 bbox 图通常是目标最清楚的一帧。
        # 成功时直接作为 final verifier 和 relation verifier 的证据，
        # 避免后续几何 stop 流程丢失“book on shelf”这类上下文。
        self.last_check_again_evidence = {
            "current_rgb_with_bbox_path": self.last_check_again_image_path,
            "object_crop_path": "",
            "centered_view_path": self.last_check_again_image_path,
            "geometry": geometry_for_evidence,
            "view_quality_facts": self._view_quality_facts(geometry_for_evidence),
            "nearby_objects": self._nearby_objects_for_final_evidence(),
            "room_context": None,
            "relation_evidence_paths": [self.last_check_again_image_path] if flag else [],
            "check_again": {
                "bbox_verified": bool(flag),
                "semantic_decision_deferred": bool(instruction_mode),
                "candidate": candidate.as_dict(),
            },
        }
        return flag

    def _raw_instruction_for_verifier(self):
        plan = getattr(self.mapper, "instruction_plan", None)
        if plan is not None:
            return str(getattr(plan, "raw_instruction", "") or self.mapper.target)
        spec = getattr(self.mapper, "instruction_spec", None)
        if spec is not None:
            return str(getattr(spec, "raw_instruction", "") or self.mapper.target)
        return str(self.instruct_goal or self.mapper.target or "")

    def _nearby_objects_for_final_evidence(self):
        nearby_objects = []
        for obj in getattr(self.mapper, "objects", []) or []:
            if obj is self.object_final:
                continue
            try:
                dist = float(np.linalg.norm(np.array(obj.position[:2]) - np.array(self.object_final.position[:2])))
            except Exception:
                continue
            if dist <= 2.0:
                nearby_objects.append({
                    "label": str(getattr(obj, "tag", "")),
                    "distance_to_candidate": round(dist, 3),
                    "position": [float(x) for x in np.array(getattr(obj, "position", []), dtype=float).reshape(-1)[:3].tolist()],
                })
        return sorted(nearby_objects, key=lambda x: x["distance_to_candidate"])[:8]

    def _build_final_verifier_evidence(self, candidate, bbox, geometry):
        episode_idx = self.episode_samples - 1
        out_dir = f'{self.save_dir}/episode-{episode_idx}/final_verifier'
        os.makedirs(out_dir, exist_ok=True)

        img = self.rgb_trajectory[-1].copy()
        img_vis = visualize_mask(img, bbox)
        current_path = os.path.join(out_dir, f'current_bbox_{self.episode_steps}.jpg')
        cv2.imwrite(current_path, img_vis)

        bbox_np = bbox.squeeze(0).cpu().numpy().astype(int)
        x1, y1, x2, y2 = bbox_np.tolist()
        pad = max(8, int(max(x2 - x1, y2 - y1) * 0.25))
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(img.shape[1] - 1, x2 + pad)
        y2 = min(img.shape[0] - 1, y2 + pad)
        crop_path = ""
        if x2 > x1 and y2 > y1:
            crop = img[y1:y2, x1:x2]
            crop_path = os.path.join(out_dir, f'object_crop_{self.episode_steps}.jpg')
            cv2.imwrite(crop_path, crop)

        positions = self.object_final.pcd.point.positions.cpu().numpy()
        distance = float(np.min(np.linalg.norm(positions[:, :2] - self.mapper.current_position[:2], axis=1)))
        geometry = dict(geometry or {})
        geometry["distance_to_object"] = distance

        evidence = {
            "current_rgb_with_bbox_path": current_path,
            "object_crop_path": crop_path,
            "centered_view_path": current_path,
            "geometry": geometry,
            "view_quality_facts": self._view_quality_facts(geometry),
            "nearby_objects": self._nearby_objects_for_final_evidence(),
            "room_context": None,
            "relation_evidence_paths": [],
        }
        with open(os.path.join(out_dir, f'evidence_{self.episode_steps}.json'), 'w', encoding='utf-8') as f:
            json.dump({
                "candidate": candidate.as_dict(),
                "raw_instruction": self._raw_instruction_for_verifier(),
                "evidence": evidence,
            }, f, ensure_ascii=False, indent=2, sort_keys=True)
        return evidence

    def _attach_view_control_context(self, evidence, candidate):
        evidence = dict(evidence or {})
        if getattr(self.view_control_state, "active", False):
            context = self.view_control_state.as_context()
            context["candidate_uid_matches_current"] = self.view_control_state.candidate_uid == candidate.uid
            context["current_candidate_uid"] = candidate.uid
            evidence["view_control"] = context
        else:
            state = getattr(self.mapper, "instruction_execution_state", None)
            pending = dict(getattr(state, "pending_verified_pair", {}) or {})
            if getattr(state, "mode", "") == "better_view_for_verified_pair" and pending:
                evidence["view_control"] = {
                    "active": True,
                    "source": "execution_pending_verified_pair",
                    "candidate_uid": pending.get("candidate_uid", ""),
                    "candidate_uid_matches_current": pending.get("candidate_uid", "") == candidate.uid,
                    "current_candidate_uid": candidate.uid,
                    "pinned_relation_context": pending.get("relation_context", {}),
                    "objective": pending.get("view_objective", {}),
                    "exhausted": False,
                }
        return evidence

    def _pin_verified_relation_from_evidence(self, evidence):
        context = dict((evidence or {}).get("verified_relation_context") or {})
        if context and getattr(self.view_control_state, "active", False):
            self.view_control_state.pin_relation_context(context)

    def _view_control_override_if_needed(self, candidate, evidence, result):
        if not getattr(self.view_control_state, "active", False):
            return result
        self.view_control_state.record_attempt(
            step=self.episode_steps,
            evidence=evidence,
            decision=result.decision,
        )
        if not (result.satisfied and result.decision == "accept"):
            return result
        if not self.view_control_state.should_block_accept(evidence):
            self.view_control_state.reset()
            return result

        # 中文说明：如果 verifier 在 better-view 子目标中想 accept，但当前
        # 证据相对 baseline 没有实质改善，且还有可行视角未尝试，则继续执行
        # view objective。这里不判断对象类别，只约束执行闭环不能“一步 retry”
        # 后提前停。
        improved = self.view_control_state.current_improvement(evidence)
        context = self.view_control_state.as_context()
        blocked = VerificationResult(
            satisfied=False,
            decision="need_better_view",
            confidence=result.confidence,
            semantic_satisfied=result.semantic_satisfied,
            view_sufficient_for_stop=False,
            satisfied_constraints=list(result.satisfied_constraints or []),
            failed_constraints=list(result.failed_constraints or []) + [
                "view-control evidence did not improve enough over baseline"
            ],
            view_feedback=(
                "The semantic relation may be satisfied, but the better-view subgoal "
                "has not produced a substantially improved final evidence view yet."
            ),
            preferred_view_goal=result.preferred_view_goal or "Try the next feasible viewpoint from the current view objective.",
            view_objective=dict(result.view_objective or context.get("objective") or {}),
            reason=(
                f"View-control blocked final accept: improvement={improved:.3f}, "
                f"required={self.view_control_state.min_required_improvement():.3f}, "
                f"remaining_proposals={self.view_control_state.remaining_count()}."
            ),
            diagnostics={
                **dict(result.diagnostics or {}),
                "view_control_override": True,
                "view_control": context,
            },
        )
        return blocked

    def _enter_better_view_control(self, candidate, evidence, result):
        objective = dict(result.view_objective or {})
        if result.preferred_view_goal and "preferred_view_goal" not in objective:
            objective["preferred_view_goal"] = result.preferred_view_goal
        if result.view_feedback and "view_feedback" not in objective:
            objective["view_feedback"] = result.view_feedback
        self.view_control_state.start(candidate.uid, objective, evidence)
        self._pin_verified_relation_from_evidence(evidence)
        self.whether_to_check_again()
        if not self.need_check_again:
            logger.info("View-control could not find a feasible better-view proposal.")
            return False
        self.waypoint = self.check_again_postion.copy()
        self.waypoint[2] = self.mapper.current_position[2]
        self.path = np.array([self.waypoint])
        self.path_index = 0
        return True

    def _defer_initial_accept_if_better_view_exists(self, candidate, evidence, result, instruction_mode=False):
        if not instruction_mode:
            return result
        if not (result.satisfied and result.decision == "accept"):
            return result
        if getattr(self.view_control_state, "active", False):
            return result

        objective = dict(result.view_objective or {})
        if not objective:
            objective = {
                "keep_visible_roles": ["candidate"],
                "improve_goals": ["clarity", "centering", "scale"],
                "minimum_expected_improvement": "moderate",
                "accept_if_no_better_view": True,
                "reason": "Initial accept should still check whether a clearly better final evidence view is feasible.",
            }
        self.view_control_state.start(candidate.uid, objective, evidence)
        self._pin_verified_relation_from_evidence(evidence)
        self.whether_to_check_again()
        if not self.need_check_again:
            self.view_control_state.reset()
            return result

        selected = None
        if self.view_control_state.last_selected_index is not None:
            try:
                selected = self.view_control_state.proposals[self.view_control_state.last_selected_index]
            except Exception:
                selected = None
        baseline_score = self.view_control_state.baseline_quality.get("score", 0.0)
        predicted_score = selected.score if selected is not None else baseline_score
        predicted_improvement = predicted_score - baseline_score
        if predicted_improvement < self.view_control_state.min_required_improvement():
            # 没有明显更好的几何视角时，不因为控制层保守性阻止 VLM 的 accept。
            self.view_control_state.reset()
            self.need_check_again = False
            return result

        self.waypoint = self.check_again_postion.copy()
        self.waypoint[2] = self.mapper.current_position[2]
        self.path = np.array([self.waypoint])
        self.path_index = 0
        return VerificationResult(
            satisfied=False,
            decision="need_better_view",
            confidence=result.confidence,
            semantic_satisfied=result.semantic_satisfied,
            view_sufficient_for_stop=False,
            satisfied_constraints=list(result.satisfied_constraints or []),
            failed_constraints=list(result.failed_constraints or []) + [
                "initial accept deferred because a better final evidence viewpoint is feasible"
            ],
            view_feedback="A better final evidence viewpoint is geometrically feasible before stopping.",
            preferred_view_goal=result.preferred_view_goal or "Move to the proposed viewpoint and re-check final instruction satisfaction.",
            view_objective=objective,
            reason=(
                f"Initial accept deferred by view-control: predicted_improvement={predicted_improvement:.3f}, "
                f"required={self.view_control_state.min_required_improvement():.3f}."
            ),
            diagnostics={
                **dict(result.diagnostics or {}),
                "view_control_initial_deferral": True,
                "view_control": self.view_control_state.as_context(),
            },
        )

    def final_instruction_check(self, evidence_override=None):
        plan = getattr(self.mapper, "instruction_plan", None)
        spec = getattr(self.mapper, "instruction_spec", None)
        if plan is None and spec is None:
            # 中文说明：普通 HM3D ObjectNav benchmark 不启用指令解析层。
            # 此时 final verifier 完全旁路，保持 STRIVE 原始 stop 行为。
            return True

        if getattr(self.object_final, "_instruction_reference_role", "") == "anchor":
            # anchor-first 模式下，anchor 只是局部搜索参考点，
            # 不能进入 final verifier，也不能被当成任务成功。到达后屏蔽
            # 该 anchor 实例，后续探索会寻找 terminal target 或其它 anchor。
            raw_instruction = self._raw_instruction_for_verifier()
            concept_id = getattr(self.object_final, "_instruction_anchor_concept_id", "")
            anchor_uid = getattr(self.object_final, "_instruction_anchor_candidate_uid", "")
            self.mapper.anchor_search_ledger.mark(
                raw_instruction=raw_instruction,
                concept_id=concept_id,
                anchor_uid=anchor_uid,
                status="searched_no_terminal_found",
                step=self.episode_steps,
                reason="Reached anchor reference; no terminal target was accepted before final stop.",
                evidence={"role": "anchor_reference"},
            )
            self.mapper.instruction_constraint_evaluator.dump_state(
                mapper=self.mapper,
                episode_idx=self.episode_samples - 1,
                step=self.episode_steps,
            )
            logger.info("Anchor reference reached and blocked for this instruction: {}", anchor_uid)
            return False

        raw_instruction = self._raw_instruction_for_verifier()
        candidate = candidate_from_object(
            self.object_final,
            canonical_label=getattr(self.mapper, "target", ""),
            step=self.episode_steps,
        )
        target_for_candidate = None
        if plan is not None:
            target_for_candidate = self.mapper.instruction_constraint_evaluator.target_for_candidate(
                self.mapper,
                plan,
                candidate,
            )
        if self.mapper.verification_ledger.is_hard_rejected(raw_instruction, candidate.uid):
            logger.info("Final verifier skips hard-rejected candidate: {}", candidate.uid)
            return False

        bbox, geometry = self._project_object_bbox_on_current_view()
        constraint_eval = None
        if evidence_override is not None:
            evidence = dict(evidence_override)
            evidence.setdefault("relation_evidence_paths", [])
            if plan is not None and target_for_candidate is not None:
                evidence = self._attach_view_control_context(evidence, candidate)
                constraint_eval = self.mapper.instruction_constraint_evaluator.evaluate_before_final_verifier(
                    mapper=self.mapper,
                    plan=plan,
                    target=target_for_candidate,
                    candidate=candidate,
                    candidate_obj=self.object_final,
                    evidence=evidence,
                    step=self.episode_steps,
                )
                evidence = constraint_eval.evidence
            if constraint_eval is not None and not constraint_eval.satisfied:
                result = VerificationResult(
                    satisfied=False,
                    decision="need_relation_check" if constraint_eval.decision == "need_relation_check" else "uncertain",
                    confidence=constraint_eval.confidence,
                    satisfied_constraints=constraint_eval.satisfied_constraints,
                    failed_constraints=constraint_eval.failed_constraints,
                    reason=constraint_eval.reason,
                    diagnostics={"constraint_eval": constraint_eval.as_dict()},
                )
            else:
                evidence = self._attach_view_control_context(evidence, candidate)
                result = self.final_instruction_verifier.verify(
                    raw_instruction=raw_instruction,
                    instruction_plan=plan,
                    candidate=candidate,
                    evidence=evidence,
                )
            evidence_paths = [
                path for path in [
                    evidence.get("current_rgb_with_bbox_path"),
                    evidence.get("object_crop_path"),
                    evidence.get("centered_view_path"),
                ] if path
            ]
        elif bbox is None:
            # final stop 姿态可能看不到点云投影，但 check_again
            # 刚刚保存过 VLM 复核图。此时继续用 check_again 图做原始指令
            # 满足度判断，而不是直接让控制流重新 stop。
            fallback_path = self.last_check_again_image_path
            evidence = {
                "current_rgb_with_bbox_path": fallback_path,
                "object_crop_path": "",
                "centered_view_path": fallback_path,
                "geometry": {
                    "projection_failed_in_final_view": True,
                },
                "view_quality_facts": self._view_quality_facts({"projection_failed_in_final_view": True}),
                "nearby_objects": [],
                "room_context": None,
                "relation_evidence_paths": [],
            }
            if fallback_path and os.path.exists(fallback_path):
                if plan is not None and target_for_candidate is not None:
                    evidence = self._attach_view_control_context(evidence, candidate)
                    constraint_eval = self.mapper.instruction_constraint_evaluator.evaluate_before_final_verifier(
                        mapper=self.mapper,
                        plan=plan,
                        target=target_for_candidate,
                        candidate=candidate,
                        candidate_obj=self.object_final,
                        evidence=evidence,
                        step=self.episode_steps,
                    )
                    evidence = constraint_eval.evidence
                    if not constraint_eval.satisfied:
                        result = VerificationResult(
                            satisfied=False,
                            decision="need_relation_check" if constraint_eval.decision == "need_relation_check" else "uncertain",
                            confidence=constraint_eval.confidence,
                            satisfied_constraints=constraint_eval.satisfied_constraints,
                            failed_constraints=constraint_eval.failed_constraints,
                            reason=constraint_eval.reason,
                            diagnostics={"constraint_eval": constraint_eval.as_dict()},
                        )
                        evidence_paths = [fallback_path]
                    else:
                        evidence = self._attach_view_control_context(evidence, candidate)
                        result = self.final_instruction_verifier.verify(
                            raw_instruction=raw_instruction,
                            instruction_plan=plan,
                            candidate=candidate,
                            evidence=evidence,
                        )
                        evidence_paths = [fallback_path]
                else:
                    evidence = self._attach_view_control_context(evidence, candidate)
                    result = self.final_instruction_verifier.verify(
                        raw_instruction=raw_instruction,
                        instruction_plan=plan,
                        candidate=candidate,
                        evidence=evidence,
                    )
                    evidence_paths = [fallback_path]
            else:
                result = self.final_instruction_verifier._fallback("object_not_projected_in_current_view")
                result.satisfied = False
                result.decision = "need_better_view"
                result.reason = "The candidate object is not visible in the current camera view."
                evidence_paths = []
        else:
            evidence = self._build_final_verifier_evidence(candidate, bbox, geometry)
            if plan is not None and target_for_candidate is not None:
                evidence = self._attach_view_control_context(evidence, candidate)
                constraint_eval = self.mapper.instruction_constraint_evaluator.evaluate_before_final_verifier(
                    mapper=self.mapper,
                    plan=plan,
                    target=target_for_candidate,
                    candidate=candidate,
                    candidate_obj=self.object_final,
                    evidence=evidence,
                    step=self.episode_steps,
                )
                evidence = constraint_eval.evidence
            if constraint_eval is not None and not constraint_eval.satisfied:
                result = VerificationResult(
                    satisfied=False,
                    decision="need_relation_check" if constraint_eval.decision == "need_relation_check" else "uncertain",
                    confidence=constraint_eval.confidence,
                    satisfied_constraints=constraint_eval.satisfied_constraints,
                    failed_constraints=constraint_eval.failed_constraints,
                    reason=constraint_eval.reason,
                    diagnostics={"constraint_eval": constraint_eval.as_dict()},
                )
            else:
                evidence = self._attach_view_control_context(evidence, candidate)
                result = self.final_instruction_verifier.verify(
                    raw_instruction=raw_instruction,
                    instruction_plan=plan,
                    candidate=candidate,
                    evidence=evidence,
                )
            evidence_paths = [
                path for path in [
                    evidence.get("current_rgb_with_bbox_path"),
                    evidence.get("object_crop_path"),
                    evidence.get("centered_view_path"),
                ] if path
            ]

        instruction_mode = plan is not None or spec is not None
        result = self._view_control_override_if_needed(candidate, evidence, result)
        result = self._defer_initial_accept_if_better_view_exists(
            candidate,
            evidence,
            result,
            instruction_mode=instruction_mode,
        )

        out_dir = f'{self.save_dir}/episode-{self.episode_samples - 1}/final_verifier'
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f'evidence_{self.episode_steps}.json'), 'w', encoding='utf-8') as f:
            json.dump({
                "candidate": candidate.as_dict(),
                "raw_instruction": raw_instruction,
                "evidence": evidence,
            }, f, ensure_ascii=False, indent=2, sort_keys=True)

        record = self.mapper.verification_ledger.put(
            raw_instruction,
            candidate.uid,
            result,
            step=self.episode_steps,
            evidence_paths=evidence_paths,
        )
        with open(os.path.join(out_dir, f'result_{self.episode_steps}.json'), 'w', encoding='utf-8') as f:
            json.dump({
                "candidate": candidate.as_dict(),
                "constraint_eval": constraint_eval.as_dict() if constraint_eval is not None else None,
                "record": record.as_dict(),
                "result": result.as_dict(),
                "ledger": self.mapper.verification_ledger.as_dict(),
            }, f, ensure_ascii=False, indent=2, sort_keys=True)
        self.mapper.instruction_constraint_evaluator.dump_state(
            mapper=self.mapper,
            episode_idx=self.episode_samples - 1,
            step=self.episode_steps,
        )

        logger.info("Final instruction verifier decision: {}", result.as_dict())
        self.instruction_decision = result.decision
        if result.satisfied and result.decision == "accept":
            relation_edges = list((evidence or {}).get("relation_edges") or [])
            if plan is not None:
                task_done = self.mapper.instruction_constraint_evaluator.apply_final_result(
                    mapper=self.mapper,
                    plan=plan,
                    target=target_for_candidate,
                    candidate=candidate,
                    result=result,
                    evidence=evidence,
                )
                if not task_done:
                    logger.info("Instruction subgoal accepted but full task is not complete yet.")
                    return False
            self.instruction_success = True
            self.instruction_decision = result.decision
            self.instruction_accept_step = self.episode_steps
            self.accepted_candidate_uid = candidate.uid
            self.accepted_relation_edge = relation_edges[0] if relation_edges else {}
            return True

        if plan is not None:
            self.mapper.instruction_constraint_evaluator.apply_final_result(
                mapper=self.mapper,
                plan=plan,
                target=target_for_candidate,
                candidate=candidate,
                result=result,
                evidence=evidence,
            )

        if result.decision == "need_better_view":
            logger.info(
                "Final verifier requests better view for {}: {} / {}",
                candidate.uid,
                result.view_feedback,
                result.preferred_view_goal,
            )
            if (result.diagnostics or {}).get("view_control_initial_deferral"):
                return False
            if self._enter_better_view_control(candidate, evidence, result):
                return False

        return False

    def step_mod(self, idx):
        self.final_instruction_accepted_this_step = False
        if self.episode_steps == 499:
            self.obs = self.env.step(0)
            self.update_trajectory()
            logger.info('Episode over!!!!!')
            return False

        pid_waypoint = self.waypoint + self.mapper.initial_position
        pid_waypoint = np.array(
            [pid_waypoint[0],
             self.env.sim.get_agent_state().position[1], pid_waypoint[1]])

        current_position = self.mapper.current_position + self.mapper.initial_position
        current_position = np.array([
            current_position[0], self.mapper.initial_position[2] - 0.88, current_position[1]
        ])
        geo_distance = self.env.sim.geodesic_distance(current_position, pid_waypoint)
        logger.info(f'Geo distance: {geo_distance}')

        # tmp = habitat_translation(self.obs['gps'])-self.mapper.initial_position
        # print(self.obs['gps'])
        # print(self.mapper.initial_position)
        # print(self.mapper.current_position)
        if self.found_goal:
            logger.info("Found goal!!!")
            logger.info("Current position: {}", self.mapper.current_position)
            logger.info("Goal position: {}", self.object_final.position)

            positions = self.object_final.pcd.point.positions.cpu().numpy()
            to_target_distance = np.min(
                np.linalg.norm(positions[:, :2] - self.mapper.current_position[:2], axis=1))
            logger.info("Distance to goal: {}", to_target_distance)

            if self.need_check_again:
                to_check_again_distance = self.calculate_geo_distance(self.check_again_postion, self.mapper.current_position)
                if to_check_again_distance > 0.5:
                    act = self.planner.get_next_action(pid_waypoint)
                else:
                    # check again
                    self.need_check_again = False

                    final_pos = self.object_final.position[:2]
                    nw_pos = self.mapper.current_position[:2]
                    nw_ori = self.rotation
                    nw_ori = habitat_rotation(nw_ori)
                    nw_ori = np.array([-nw_ori[0, 2], -nw_ori[1, 2]])
                    final_pos = final_pos - nw_pos

                    nw_ori = nw_ori / np.linalg.norm(nw_ori)
                    final_pos = final_pos / np.linalg.norm(final_pos)
                    dot_product = np.clip(np.dot(nw_ori, final_pos), -1.0, 1.0)
                    angle_rad = np.arccos(dot_product)
                    angle_deg = np.degrees(angle_rad)

                    cross_product = np.cross(nw_ori, final_pos)
                    direction = 3 if cross_product > 0 else 2
                    num_turns = int(np.round(angle_deg / 30))

                    logger.info("Now step: {}", self.episode_steps)
                    logger.info("Direction: {}", direction)
                    logger.info("Num turns: {}", num_turns)

                    for i in range(num_turns):
                        self.obs = self.env.step(direction)
                        self.update_trajectory(self.on_node_flag)
                        if self.env.episode_over:
                            return False

                    self.found_goal = self.check_again(self.episode_steps)
                    logger.info("Check again at step: {}", self.episode_steps)
                    logger.info("Check again: {}", self.found_goal)

                    if not self.found_goal:
                        self.after_check_again()
                    else:
                        final_instruction_flag = self.final_instruction_check(
                            evidence_override=self.last_check_again_evidence,
                        )
                        if final_instruction_flag:
                            self.final_instruction_accepted_this_step = True
                            act = 0
                        else:
                            pid_waypoint = self.waypoint + self.mapper.initial_position
                            pid_waypoint = np.array(
                                [pid_waypoint[0],
                                 self.env.sim.get_agent_state().position[1], pid_waypoint[1]])
                            if self.need_check_again:
                                act = self.planner.get_next_action(pid_waypoint)
                                if act == 0:
                                    self.need_check_again = False
                                    self.after_check_again()
                                    pid_waypoint = self.waypoint + self.mapper.initial_position
                                    pid_waypoint = np.array(
                                        [pid_waypoint[0],
                                         self.env.sim.get_agent_state().position[1], pid_waypoint[1]])
                            else:
                                self.after_check_again()
                                pid_waypoint = self.waypoint + self.mapper.initial_position
                                pid_waypoint = np.array(
                                    [pid_waypoint[0],
                                     self.env.sim.get_agent_state().position[1], pid_waypoint[1]])
                            act = self.planner.get_next_action(pid_waypoint)
            else:
                if to_target_distance > self.success_distance * self.stop_criterion:
                    act = self.planner.get_next_action(pid_waypoint)
                else:
                    act = 0

        # not use else because self.check_again may change the self.found_goal
        if not self.found_goal:
            act = self.planner.get_next_action(pid_waypoint)
            while act == 0 and not self.found_goal:
                self.path_index += 1
                if self.path_index >= len(self.path):
                    # get to the next waypoint we want
                    if self.relocate:
                        flag, self.found_goal = self.make_plan_mod_relocate(
                            rotate=True, idx=idx, node=self.waypoint_final)
                    else:
                        flag, self.found_goal = self.make_plan_mod_no_relocate(
                            rotate=True,
                            idx=idx,
                            node=self.waypoint_final,
                            use_gpt_relocate=self.gpt_relocate,
                        )
                    if self.env.episode_over:
                        return False

                    if not flag and not self.env.episode_over:
                        self.obs = self.env.step(act)
                        self.update_trajectory(self.on_node_flag)
                        logger.info(self.env.episode_over)
                        return False
                else:
                    logger.info(f'!!!!!!!!!!!!!!!!Bug: enter make_plan_mod_process')
                    self.obs = self.env.step(0)
                    self.update_trajectory(self.on_node_flag)
                    logger.info(self.env.episode_over)
                    return False
                    # self.make_plan_mod_process(idx=idx, node=None, path_idx=self.path_index)

                self.waypoint = self.path[self.path_index]
                self.waypoint[2] = self.mapper.current_position[2]

                # self.waypoint = [-6.1704755, -1.333439, -0.8]

                logger.info(f'Waypoint: {self.waypoint}')
                pid_waypoint = self.waypoint + self.mapper.initial_position
                pid_waypoint = np.array(
                    [pid_waypoint[0],
                     self.env.sim.get_agent_state().position[1], pid_waypoint[1]])
                act = self.planner.get_next_action(pid_waypoint)

            if act == 1:
                self.on_node_flag = False

        logger.info("Step: {}", self.episode_steps)
        logger.info("Next Action: {}", act)
        logger.info("Episode over: {}", self.env.episode_over)
        logger.info("Found goal: {}", self.found_goal)
        logger.info("Waypoint: {}", self.waypoint)

        if not self.env.episode_over:
            if self.found_goal and act == 0 and not self.final_instruction_accepted_this_step:
                final_check_flag = self.final_check()
                if final_check_flag:
                    final_instruction_flag = self.final_instruction_check()
                    if not final_instruction_flag:
                        if self.need_check_again:
                            pid_waypoint = self.waypoint + self.mapper.initial_position
                            pid_waypoint = np.array(
                                [pid_waypoint[0],
                                 self.env.sim.get_agent_state().position[1], pid_waypoint[1]])
                            act = self.planner.get_next_action(pid_waypoint)
                            if act == 0:
                                # 已经没有可移动的更好视角时，不能把 act=0 当成成功 stop。
                                self.need_check_again = False
                                self.after_check_again()
                        else:
                            self.after_check_again()
                        pid_waypoint = self.waypoint + self.mapper.initial_position
                        pid_waypoint = np.array(
                            [pid_waypoint[0],
                             self.env.sim.get_agent_state().position[1], pid_waypoint[1]])
                        act = self.planner.get_next_action(pid_waypoint)
                else:
                    pid_waypoint = self.waypoint + self.mapper.initial_position
                    pid_waypoint = np.array(
                        [pid_waypoint[0],
                         self.env.sim.get_agent_state().position[1], pid_waypoint[1]])
                    act = self.planner.get_next_action(pid_waypoint)

            logger.info("Next Action: {}", act)
            self.obs = self.env.step(act)
            self.update_trajectory(self.on_node_flag)

            logger.info(self.env.episode_over)

            return True

    def to_json(self):
        return self.mapper.to_json()

    def whether_to_check_again(self):
        logger.info(f'Need to check again')
        self.need_check_again = True

        # find the best point to check
        pathfinder = self.env.sim.pathfinder
        import habitat_sim
        path_request = habitat_sim.ShortestPath()
        current_position = self.found_goal_position + self.mapper.initial_position
        current_position = np.array([
            current_position[0], self.env.sim.get_agent_state().position[1], current_position[1]
        ])
        path_request.requested_start = current_position
        pid_waypoint = self.waypoint + self.mapper.initial_position
        pid_waypoint = np.array(
            [pid_waypoint[0],
             self.env.sim.get_agent_state().position[1], pid_waypoint[1]])
        path_request.requested_end = pid_waypoint

        # 计算最短路径
        found_path = pathfinder.find_path(path_request)
        points = path_request.points
        logger.info(f'Path: {points}')
        proposals, interpolated_path = build_check_again_viewpoints(
            object_node=self.object_final,
            camera_intrinsic=self.mapper.camera_intrinsic,
            mapper_initial_position=self.mapper.initial_position,
            habitat_path_points=points,
            target_height=self.found_goal_position[2] + 0.88,
            success_distance=self.success_distance,
            stop_criterion=self.stop_criterion,
        )
        logger.info(f"Interpolated_path: {interpolated_path}")

        best_candidate = None
        for candidate in proposals:
            logger.info(
                "Check-again candidate {} score {:.3f}, visible {:.3f}, center {:.3f}, border {:.3f}, area {:.3f}, dist {:.3f}",
                candidate["position"],
                candidate["score"],
                candidate["visible_ratio"],
                candidate["center_score"],
                candidate["border_score"],
                candidate["area_ratio"],
                candidate["distance_to_target"],
            )
            if best_candidate is None or candidate["score"] > best_candidate["score"]:
                best_candidate = candidate

        current_candidate = candidate_from_object(
            self.object_final,
            canonical_label=getattr(self.mapper, "target", ""),
            step=self.episode_steps,
        )
        state = getattr(self.mapper, "instruction_execution_state", None)
        pending_pair_active = (
            getattr(state, "mode", "") == "better_view_for_verified_pair"
            and bool(getattr(state, "pending_verified_pair", {}) or {})
        )
        if getattr(self.view_control_state, "active", False) and (
            self.view_control_state.candidate_uid == current_candidate.uid
            or pending_pair_active
        ):
            self.view_control_state.set_proposals(proposals)
            proposal = self.view_control_state.next_proposal()
            if proposal is not None:
                logger.info(
                    "Selected view-control proposal: {}, score {:.3f}, remaining {}",
                    proposal.pose,
                    proposal.score,
                    self.view_control_state.remaining_count(),
                )
                self.check_again_postion = np.array(proposal.pose, dtype=float)
                self.need_check_again = True
                return
            logger.info("View-control proposals exhausted for {}", current_candidate.uid)
            self.need_check_again = False
            self.check_again_postion = self.mapper.current_position
        elif best_candidate is not None:
            logger.info(
                "Selected check-again viewpoint: {}, score {:.3f}, visible {:.3f}, center {:.3f}, border {:.3f}, area {:.3f}, dist {:.3f}",
                best_candidate["position"],
                best_candidate["score"],
                best_candidate["visible_ratio"],
                best_candidate["center_score"],
                best_candidate["border_score"],
                best_candidate["area_ratio"],
                best_candidate["distance_to_target"],
            )
            self.check_again_postion = best_candidate["position"]
        elif len(interpolated_path) > 0:
            logger.info(f"Can't score a good point to check, use the last point")
            self.check_again_postion = interpolated_path[-1]
        else:
            logger.info(f"Can't find a good point to check, use the current position")
            self.check_again_postion = self.mapper.current_position

    def after_check_again(self):
        # 中文说明：任何复核失败都必须退出“已找到目标”状态。
        # 否则 STRIVE 的 stop 分支会在后续 step 继续围绕同一候选触发，
        # 即使 final verifier 已经把该实例 hard-reject。
        if getattr(self.object_final, "_instruction_reference_role", "") == "anchor":
            self.mapper.anchor_search_ledger.mark(
                raw_instruction=self._raw_instruction_for_verifier(),
                concept_id=getattr(self.object_final, "_instruction_anchor_concept_id", ""),
                anchor_uid=getattr(self.object_final, "_instruction_anchor_candidate_uid", ""),
                status="searched_no_terminal_found",
                step=self.episode_steps,
                reason="Anchor reference local search ended without an accepted terminal target.",
                evidence={"role": "anchor_reference"},
            )
        self.view_control_state.reset()
        self.found_goal = False
        self.need_check_again = False
        old_tag = str(getattr(self.object_final, "tag", ""))
        old_confidence = getattr(self.object_final, "confidence", torch.tensor(1.0))
        # 中文说明：这里只是让该实例退出“当前目标候选”状态，不能清空
        # ObjectNode 的历史类别计数。mapper 后续 association 仍会按旧 tag
        # 读取 num_list/conf_list；如果直接替换成 {"nothing": ...} 会破坏地图记忆。
        if old_tag and old_tag not in getattr(self.object_final, "num_list", {}):
            self.object_final.num_list[old_tag] = 1
        if old_tag and old_tag not in getattr(self.object_final, "conf_list", {}):
            self.object_final.conf_list[old_tag] = old_confidence
        self.object_final.tag = "nothing"
        self.object_final.confidence = torch.tensor(1.0)
        self.object_final.conf_list["nothing"] = self.object_final.confidence
        self.object_final.num_list["nothing"] = 10000

        waypoint_node = self.mapper.explore_after_check()
        if waypoint_node is None:
            logger.info("Fully Explored!!!!!")
            return 0

        self.waypoint_final = waypoint_node
        self.waypoint = waypoint_node.position
        self.waypoint[2] = self.mapper.current_position[2]
        self.path = np.array([self.waypoint])
        self.path_index = 0

        if waypoint_node.state == 1:
            self.just_come_back = True
        else:
            self.just_come_back = False

    def final_check(self):
        final_point = self.mapper.current_position
        final_point[2] = final_point[2] + 0.88
        final_point_voxel_idx = translate_single_point_to_grid(final_point, self.mapper.grid_resolution,
                                                               self.mapper.voxel_dimension)

        voxels = np.zeros(self.mapper.voxel_dimension, dtype=np.int32)
        voxel_idxs = translate_point_to_grid(self.mapper.useful_pcd.point.positions.cpu().numpy(),
                                             self.mapper.grid_resolution, self.mapper.voxel_dimension)
        voxels[voxel_idxs[:, 0], voxel_idxs[:, 1], voxel_idxs[:, 2]] = 1

        obj_voxel_idxs = translate_point_to_grid(self.object_final.pcd.point.positions.cpu().numpy(),
                                                 self.mapper.grid_resolution, self.mapper.voxel_dimension)
        # voxels[obj_voxel_idxs[:, 0], obj_voxel_idxs[:, 1], obj_voxel_idxs[:, 2]] = 0

        big_visible_flag = False
        for obj_voxel_idx in obj_voxel_idxs:
            ray_idxs = bresenham_3d(obj_voxel_idx, final_point_voxel_idx)[1:]
            small_visible_flag = True
            for ray_idx in ray_idxs:
                if ray_idx[0] < 0 or ray_idx[0] >= self.mapper.voxel_dimension[0] or \
                        ray_idx[1] < 0 or ray_idx[1] >= self.mapper.voxel_dimension[1] or \
                        ray_idx[2] < 0 or ray_idx[2] >= self.mapper.voxel_dimension[2]:
                    continue
                if voxels[ray_idx[0], ray_idx[1], ray_idx[2]] == 1:
                    small_visible_flag = False
                    continue

            if small_visible_flag:
                big_visible_flag = True
                break

        if big_visible_flag:
            # if the object is visible, then stop
            logger.info(f"Object is visible, Robot can stop at {final_point}")
            return True
        else:
            logger.info(f"!!!!!!!!!!!Can't see the object, because of occlusion.")
            pid_waypoint = self.found_goal_position + self.mapper.initial_position
            pid_waypoint = np.array(
                [pid_waypoint[0],
                 self.env.sim.get_agent_state().position[1], pid_waypoint[1]])
            logger.info(f'First go back to the waypoint finding the object.')
            while True:
                act = self.planner.get_next_action(pid_waypoint)
                if act == 0:
                    break
                logger.info("Step: {}", self.episode_steps)
                logger.info("Next Action: {}", act)
                logger.info("Episode over: {}", self.env.episode_over)
                logger.info("Found goal: {}", self.found_goal)
                logger.info("Waypoint: {}", self.found_goal_position)

                self.obs = self.env.step(act)
                self.update_trajectory(self.on_node_flag)

            self.find_final_waypoint()

            return False


    def find_final_waypoint(self):
        waypoint_tmp = self.object_final.find_closest(self.found_goal_position)
        self.path = np.array([waypoint_tmp])
        non_valid_stop = True

        iter_num = 0
        distance_to_initial_waypoint = np.linalg.norm(waypoint_tmp[:2] - self.found_goal_position[:2])
        max_iter_num = min(8, int(distance_to_initial_waypoint / 0.1))
        while non_valid_stop and iter_num < max_iter_num:
            non_valid_stop = False
            pathfinder = self.env.sim.pathfinder

            # 设置一个 ShortestPath 查询
            import habitat_sim
            path_request = habitat_sim.ShortestPath()
            current_position = self.found_goal_position + self.mapper.initial_position
            current_position = np.array([
                current_position[0], self.mapper.initial_position[2] - 0.88, current_position[1]
            ])
            path_request.requested_start = current_position
            pid_waypoint = waypoint_tmp + self.mapper.initial_position
            pid_waypoint = np.array(
                [pid_waypoint[0],
                 self.env.sim.get_agent_state().position[1], pid_waypoint[1]])
            path_request.requested_end = pid_waypoint

            # 计算最短路径
            found_path = pathfinder.find_path(path_request)
            if found_path:
                points = path_request.points
                # logger.info(f'Path: {points}')
                points = np.array(points)
                # swithc the y and z axis
                points = np.array([points[:, 0], points[:, 2], points[:, 1]]).T
                points = points - self.mapper.initial_position
                points[:, 2] = -0.8
                logger.info(f"Path points: {points}")

                interpolated_path = []
                for i in range(len(points) - 1):
                    point1 = points[i]
                    point2 = points[i + 1]
                    # use np.linspace to interpolate the path between the two points
                    distance = np.linalg.norm(point1 - point2)
                    num_points = max(1, int(distance / 0.25))
                    interpolated_points = [(1 - t) * point1 + t * point2 for t in np.linspace(0, 1, num_points)]
                    interpolated_path.extend(interpolated_points)
                # invert the path
                interpolated_path = interpolated_path[::-1]
                # find the first point in interpolated_path that is far away from the self.obj_final from a certain distance
                positions = self.object_final.pcd.point.positions.cpu().numpy()
                final_point = None
                for point in interpolated_path:
                    to_target_distance = np.min(np.linalg.norm(positions[:, :2] - point[:2], axis=1))
                    if to_target_distance > self.success_distance * self.stop_criterion:
                        final_point = point
                        break
                if final_point is None:
                    final_point = interpolated_path[-1]

                # Compute the stop orientation so the final view faces the object.
                final_pos = self.object_final.position[:2]
                stop_pos = final_point[:2]
                final_pos = final_pos - stop_pos
                final_pos = final_pos / np.linalg.norm(final_pos)
                rotation_matrix = np.array([[-final_pos[1], 0, -final_pos[0]],
                                             [final_pos[0], 0, -final_pos[1]],
                                              [0, 1, 0]])

                final_point[2] = self.found_goal_position[2] + 0.88
                final_point_voxel_idx = translate_single_point_to_grid(final_point, self.mapper.grid_resolution, self.mapper.voxel_dimension)

                voxels = np.zeros(self.mapper.voxel_dimension, dtype=np.int32)
                voxel_idxs = translate_point_to_grid(self.mapper.useful_pcd.point.positions.cpu().numpy(), self.mapper.grid_resolution, self.mapper.voxel_dimension)
                voxels[voxel_idxs[:, 0], voxel_idxs[:, 1], voxel_idxs[:, 2]] = 1

                obj_voxel_idxs = translate_point_to_grid(self.object_final.pcd.point.positions.cpu().numpy(), self.mapper.grid_resolution, self.mapper.voxel_dimension)
                # voxels[obj_voxel_idxs[:, 0], obj_voxel_idxs[:, 1], obj_voxel_idxs[:, 2]] = 0

                big_visible_flag = False
                for obj_voxel_idx in obj_voxel_idxs:
                    ray_idxs = bresenham_3d(obj_voxel_idx, final_point_voxel_idx)[1:]
                    small_visible_flag = True
                    for ray_idx in ray_idxs:
                        if ray_idx[0] < 0 or ray_idx[0] >= self.mapper.voxel_dimension[0] or \
                                ray_idx[1] < 0 or ray_idx[1] >= self.mapper.voxel_dimension[1] or \
                                ray_idx[2] < 0 or ray_idx[2] >= self.mapper.voxel_dimension[2]:
                            continue
                        if voxels[ray_idx[0], ray_idx[1], ray_idx[2]] == 1:
                            small_visible_flag = False
                            continue

                    if small_visible_flag:
                        big_visible_flag = True
                        break

                if big_visible_flag:
                    # if the object is visible, then stop
                    non_valid_stop = False
                    logger.info(f"Object is visible, {waypoint_tmp} is a valid waypoint")
                    self.waypoint = waypoint_tmp
                    self.waypoint[2] = self.found_goal_position[2]
                    self.path = np.array([self.waypoint])
                    self.path_index = 0
                    return

                else:
                    non_valid_stop = True
                    logger.info(f"Object is not visible, {waypoint_tmp} is not a valid stop point")
                    # move the pid_waypoint closer to self.mapper.current_position
                    vector_tmp = (self.found_goal_position - waypoint_tmp) / np.linalg.norm(self.found_goal_position - waypoint_tmp)
                    waypoint_tmp = waypoint_tmp + vector_tmp * 0.1

            else:
                non_valid_stop = True
                logger.info(f"Path to {waypoint_tmp} is not found. {waypoint_tmp} is unreachable.")
                vector_tmp = (self.found_goal_position - waypoint_tmp) / np.linalg.norm(
                    self.found_goal_position - waypoint_tmp)
                waypoint_tmp = waypoint_tmp + vector_tmp * 0.1

            iter_num += 1

        logger.info(f"Strange case, can't find a valid waypoint.")
        logger.info(f"Use waypoint: {waypoint_tmp}")
        self.waypoint = waypoint_tmp
        self.waypoint[2] = self.found_goal_position[2]
        self.path = np.array([self.waypoint])
        self.path_index = 0

        return

    def calculate_geo_distance(self, point1, point2):
        point1 = point1 + self.mapper.initial_position
        point1 = np.array(
            [point1[0],
             self.env.sim.get_agent_state().position[1], point1[1]])
        point2 = point2 + self.mapper.initial_position
        point2 = np.array([
            point2[0], self.env.sim.get_agent_state().position[1], point2[1]
        ])
        geodesic_distance = self.env.sim.geodesic_distance(point1, point2)

        return geodesic_distance
