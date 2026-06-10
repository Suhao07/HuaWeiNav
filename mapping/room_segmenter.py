"""Room-level map segmentation for STRIVE mapper runtime.

This module owns the room partition algorithm. It mutates the mapper state passed
in by the compatibility wrapper, but keeps the heavy image/mask/region logic out
of ``mapper_with_process_obs.py``.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import open3d as o3d
from loguru import logger

from mapping_utils.projection import project_room, translate_grid_to_point, translate_point_to_grid
from mapping_utils.representation import Room_node

def segment_room(mapper, step_idx):
    """Partition navigation nodes into room-level regions from the map point cloud."""

    mapper.room_nodes = []
    for node in mapper.nodes:
        node.room_idx = -1

    obs_pcd = mapper.scene_pcd.select_by_index((mapper.scene_pcd.point.positions[:, 2]
                                              < mapper.ceiling_height).nonzero()[0])
    nav_pcd = mapper.navigable_pcd
    tmp_floor_path = f'{mapper.save_dir}/episode-{mapper.episode_idx}/room_inter/step_{mapper.update_iterations}'
    os.makedirs(tmp_floor_path, exist_ok=True)

    save_intermediate_results = True

    obs_pcd = obs_pcd.voxel_down_sample(voxel_size=0.05)
    nav_pcd = nav_pcd.voxel_down_sample(voxel_size=0.05)

    # floor_pcd.voxel_down_sample(voxel_size=0.05)
    xyz = obs_pcd.point.positions.cpu().numpy()
    nav_points = nav_pcd.point.positions.cpu().numpy()
    xyz = np.concatenate([xyz, nav_points], axis=0)

    # print(xyz)
    xyz_full = xyz.copy()
    floor_zero_level = -0.8
    floor_height = 0.8
    ## Slice below the ceiling ##
    xyz = xyz[xyz[:, 2] < floor_height]
    xyz = xyz[xyz[:, 2] >= floor_zero_level + 0.2]
    xyz_full = xyz_full[xyz_full[:, 2] < floor_height]

    # 房间分割只在 2D 地图上做：障碍/墙体提供分割边界，nav 点提供
    # 可通行区域。三维语义对象不参与 room id 生成。
    # project the point cloud to 2d
    # pcd_2d = xyz[:, [0, 1]]
    # xyz_full = xyz_full[:, [0, 1]]
    pcd_2d = xyz
    xyz_full = xyz_full

    room_resolution = 0.05
    # room_dimension = mapper.voxel_dimension
    # room_dimension[0] *= (mapper.grid_resolution // room_resolution)
    # room_dimension[1] *= (mapper.grid_resolution // room_resolution)

    room_dimension = np.asarray([1000,1000,20])
    nodes_positions = translate_point_to_grid(
        mapper.get_nodes_positions(),
        grid_resolution=room_resolution,
        voxel_dimension=room_dimension,
    )[:, :2]
    nodes_positions[:, 0] = np.clip(nodes_positions[:, 0], 0, room_dimension[0] - 1)
    nodes_positions[:, 1] = np.clip(nodes_positions[:, 1], 0, room_dimension[1] - 1)

    hist = project_room(pcd_2d,grid_resolution=room_resolution,voxel_dimension=room_dimension)
    if save_intermediate_results:
        cv2.imwrite(os.path.join(tmp_floor_path, "obstacale_2D_histogram.png"), hist)

    # 用障碍点密度近似墙体骨架。早期地图稀疏时该结果可能为空，
    # 因此后面必须保留 fallback_single_room。
    # applythresholding
    hist = cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    hist = cv2.GaussianBlur(hist, (5, 5), 1)
    hist_threshold = 0.25 * np.max(hist)
    _, walls_skeleton_hist = cv2.threshold(hist, hist_threshold, 255, cv2.THRESH_BINARY)

    # apply closing to the walls skeleton
    kernal = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    walls_skeleton_hist = cv2.morphologyEx(walls_skeleton_hist,
                                           cv2.MORPH_CLOSE,
                                           kernal,
                                           iterations=1)

    if save_intermediate_results:
        cv2.imwrite(os.path.join(tmp_floor_path, "walls_skeleton.png"), walls_skeleton_hist)

    # extract outside boundary from histogram of xyz_full
    hist_full = project_room(xyz_full,grid_resolution=room_resolution,voxel_dimension=room_dimension)

    hist_full = cv2.normalize(hist_full, hist_full, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, outside_boundary = cv2.threshold(hist_full, 0.001, 255, cv2.THRESH_BINARY)
    outside_boundary = outside_boundary.astype(np.uint8)

    # draw contours fill the blank
    contours, _ = cv2.findContours(outside_boundary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outside_boundary = np.zeros_like(outside_boundary)
    cv2.drawContours(outside_boundary, contours, -1, (255, 255, 255), -1)

    # visualize the walls_skeleton_hist
    if save_intermediate_results:
        cv2.imwrite(os.path.join(tmp_floor_path, "full_map1.png"), outside_boundary)

        # save the full map as point cloud
        positions = np.where(outside_boundary != 0)
        # switch the first 2 dimension of the positions
        positions = np.asarray(positions).T
        position_world = translate_grid_to_point(np.asarray(positions), grid_resolution=room_resolution,
                                                 voxel_dimension=room_dimension)
        # save the pcd
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(position_world)
        o3d.io.write_point_cloud(os.path.join(tmp_floor_path, f"full_map_ori.ply"), pcd)

    # print(outside_boundary.shape, walls_skeleton_hist.shape)
    wall_positions = np.where(walls_skeleton_hist != 0)
    outside_boundary[wall_positions] = 0

    if save_intermediate_results:
        cv2.imwrite(os.path.join(tmp_floor_path, "full_map2.png"), outside_boundary)

    # draw the outside boundary
    outside_boundary_tmp = cv2.bitwise_not(outside_boundary)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        outside_boundary_tmp, connectivity=8)

    # get all components with the area larger than 10
    outside_boundary_tmp = np.zeros_like(outside_boundary_tmp)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > 10:
            outside_boundary_tmp[labels == i] = 255

    outside_boundary_tmp = cv2.bitwise_not(outside_boundary_tmp)

    # visualize the largest component
    if save_intermediate_results:
        cv2.imwrite(os.path.join(tmp_floor_path, "full_map3.png"), outside_boundary_tmp)

    # get the contours of the outside boundary
    contours, _ = cv2.findContours(outside_boundary_tmp, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    outside_boundary_tmp = np.zeros_like(outside_boundary_tmp)
    cv2.drawContours(outside_boundary_tmp, contours, -1, (255, 255, 255), -1)
    outside_boundary_tmp = outside_boundary_tmp.astype(np.uint8)

    # get the components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        outside_boundary_tmp, connectivity=8)
    if num_labels <= 1:
        logger.warning("Room segmentation found no connected free-space component at step {}.", step_idx)
        fallback_single_room(mapper, nodes_positions)
        return
    # get the largest component
    max_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    outside_boundary_tmp = np.zeros_like(outside_boundary_tmp)
    outside_boundary_tmp[labels == max_label] = 255

    # apply closing to the outside boundary
    outside_boundary_tmp = cv2.bitwise_not(outside_boundary_tmp)
    kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    outside_boundary_tmp = cv2.morphologyEx(outside_boundary_tmp,
                                            cv2.MORPH_CLOSE,
                                            kernal,
                                            iterations=2)
    outside_boundary_tmp = cv2.bitwise_not(outside_boundary_tmp)

    if save_intermediate_results:
        cv2.imwrite(os.path.join(tmp_floor_path, "full_map4.png"), outside_boundary_tmp)
        #

    # connected component markers 是最终 room region 的来源。
    # 这里不用 watershed 的 -1 边界语义，避免 markers 未初始化导致崩溃。
    free_mask = (outside_boundary_tmp > 0).astype(np.uint8) * 255
    num_regions, markers, region_stats, _ = cv2.connectedComponentsWithStats(
        free_mask, connectivity=8
    )
    if num_regions <= 1:
        logger.warning("Room segmentation produced no region markers at step {}.", step_idx)
        fallback_single_room(mapper, nodes_positions)
        return

    # 遍历每个唯一的 region_id，并创建 mask
    unique_regions = np.unique(markers)
    region_masks = {}
    position_worlds = {}
    position_maps = {}
    for region_id in unique_regions:
        if region_id in [-1, 0]:  # 跳过 watershed 线和背景
            continue
        if region_stats[int(region_id), cv2.CC_STAT_AREA] <= 10:
            continue
        mask = (markers == region_id).astype(np.uint8) * 255
        region_masks[region_id] = mask  # 存储每个区域的 mask
        # transform to world point cloud
        positions = np.where(mask == 255)
        positions = np.asarray(positions).T
        position_world = translate_grid_to_point(np.asarray(positions),grid_resolution=room_resolution,voxel_dimension=room_dimension)
        position_map = translate_point_to_grid(position_world,grid_resolution=mapper.grid_resolution,voxel_dimension=mapper.voxel_dimension)[:, :2]
        position_worlds[region_id] = position_world
        position_maps[region_id] = position_map

        # determine which points belong to this room region
        nodes_in_region = []
        for node_idx, nodes_position in enumerate(nodes_positions):
            if mask[nodes_position[0], nodes_position[1]] == 255:
                nodes_in_region.append(mapper.nodes[node_idx])

        if len(nodes_in_region) != 0:
            # room_node = Room_node(nodes_in_region, position_world, len(mapper.room_nodes))
            for node in nodes_in_region:
                node.room_idx = region_id

            # # save the pcd
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(position_world)
            # o3d.io.write_point_cloud(os.path.join(tmp_floor_path, f"region_{len(mapper.room_nodes)}.ply"), pcd)

            # mapper.room_nodes.append(room_node)
    for node in mapper.nodes:
        room_idx = node.room_idx
        node_idx = node.idx
        current_position = nodes_positions[node_idx]
        if room_idx == -1:
            # 有些 node 可能落在墙体膨胀或 mask 边缘外。此时归到最近
            # region，保证每个 viewpoint 都有 room_idx 可用于探索策略。
            # find the closest mask to this node
            node_distance_to_mask = {}
            for region_id, mask in region_masks.items():
                positions = np.where(mask == 255)
                distance = np.linalg.norm(np.array(positions).T - current_position, axis=1)
                distance_min = np.min(distance)
                node_distance_to_mask[region_id] = distance_min
            if not node_distance_to_mask:
                fallback_single_room(mapper, nodes_positions)
                return
            closest_region_id = min(node_distance_to_mask, key=node_distance_to_mask.get)
            node.room_idx = closest_region_id

    remaining_node_idxs = [node.idx for node in mapper.nodes]

    while remaining_node_idxs:
        node_idx = remaining_node_idxs[0]  # 取出当前节点
        node = mapper.nodes[node_idx]
        room_idx = node.room_idx

        # 获取该 room 内的所有节点
        nodes_in_region = [mapper.nodes[idx] for idx in remaining_node_idxs if mapper.nodes[idx].room_idx == room_idx]
        node_idxs_in_region = [node.idx for node in nodes_in_region]
        room_node = Room_node(nodes_in_region, position_worlds[room_idx], position_maps[room_idx], len(mapper.room_nodes))
        logger.info(f"Room {len(mapper.room_nodes)} has Nodes: {node_idxs_in_region}")

        for node_in_region in nodes_in_region:
            node_in_region.room_idx = len(mapper.room_nodes)

        mapper.room_nodes.append(room_node)

        # 移除已处理的节点
        remaining_node_idxs = list(set(remaining_node_idxs) - set(node_idxs_in_region))

    # room distance 用当前图上的最近 frontier node 估计，供 relocation
    # 和 deterministic room policy 排序使用。
    # update the distance from current pos to each room
    current_node = mapper.nodes[mapper.current_node_idx]
    for room_node in mapper.room_nodes:
        closet_node = mapper.find_closet_viewpoint_in_room(room_node)
        if closet_node is None:
            continue

        path_length = mapper.get_path_length(closet_node)
        room_node.distance = path_length


    # 区分 room 内 frontier 与通向房间外的 frontier：
    # grid_map=1 表示室内继续探索，grid_map=2 更像跨房间出口。
    # update the frontier state(frontier in room and frontier out room)
    # first get all the room mask
    room_map = np.ones((mapper.voxel_dimension[0], mapper.voxel_dimension[1]))
    for room_node in mapper.room_nodes:
        room_map_pos = room_node.mask_map
        room_map[room_map_pos[:, 0], room_map_pos[:, 1]] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    room_map = cv2.dilate(room_map, kernel, iterations=2)

    for frontier_idxs in mapper.current_global_frontier_map_idxs:
        # boarder_frontier = [[x,y] for (x,y) in frontier_idxs if room_map[x, y] == 1]
        inner_frontier = [[x,y] for (x,y) in frontier_idxs if room_map[x, y] == 0]

        if len(inner_frontier) > 0.6 * len(frontier_idxs):
            # print(inner_frontier)
            # print(frontier_idxs)
            # print(len(inner_frontier), len(frontier_idxs))
            mapper.grid_map[frontier_idxs[:, 0], frontier_idxs[:, 1]] = 1
        else:
            mapper.grid_map[frontier_idxs[:, 0], frontier_idxs[:, 1]] = 2

def fallback_single_room(mapper, nodes_positions=None):
    """Fallback room partition when watershed/connected components are empty.

    早期探索阶段点云和墙体骨架可能还不足以切出稳定房间。此时不能让
    relocation 崩溃，先把已有 viewpoint 归为一个临时 room 后续地图更完整
    时 `segment_room()` 会重新生成更细的 room nodes。
    """

    if len(mapper.nodes) == 0:
        mapper.room_nodes = []
        return
    position_world = np.asarray(mapper.get_nodes_positions(), dtype=float).reshape(-1, 3)
    position_map = translate_point_to_grid(
        position_world,
        grid_resolution=mapper.grid_resolution,
        voxel_dimension=mapper.voxel_dimension,
    )[:, :2]
    position_map[:, 0] = np.clip(position_map[:, 0], 0, mapper.voxel_dimension[0] - 1)
    position_map[:, 1] = np.clip(position_map[:, 1], 0, mapper.voxel_dimension[1] - 1)
    room_node = Room_node(list(mapper.nodes), position_world, position_map, len(mapper.room_nodes))
    for node in mapper.nodes:
        node.room_idx = room_node.room_id
    closet_node = mapper.find_closet_viewpoint_in_room(room_node)
    if closet_node is not None:
        room_node.distance = mapper.get_path_length(closet_node)
    mapper.room_nodes.append(room_node)

