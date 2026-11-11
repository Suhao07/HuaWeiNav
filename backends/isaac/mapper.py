"""Original STRIVE mapping with explicit Isaac camera and XY-query boundaries."""
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image


class IsaacMapperQueries:
    """Project queries onto freemap XY; never certify three-dimensional support."""

    domain = "freemap_xy"

    def __init__(self, client, camera_position):
        self.client = client
        self.camera_position = np.array(camera_position, dtype=float, copy=True)

    def _query(self, action, field, values, batch_size):
        rows = []
        for offset in range(0, len(values), batch_size):
            batch = values[offset:offset + batch_size]
            response = self.client.command(action, **{field: batch})
            if (not isinstance(response, dict) or response.get("geometry_domain") != self.domain
                    or not isinstance(response.get("results"), list)
                    or len(response["results"]) != len(batch)):
                raise ValueError("Invalid freemap query response")
            rows.extend(response["results"])
        return rows

    def navigable_mask(self, positions):
        points = np.asarray(positions, dtype=float)
        if points.shape == (0,):
            points = points.reshape(0, 3)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError('Expected an N by 3 point array')
        points = points[:, :2].tolist()
        rows = self._query("walkability_xy", "positions", points, 256)
        for point, row in zip(points, rows):
            if (not isinstance(row, dict) or row.get("position") != point
                    or type(row.get("walkable")) is not bool):
                raise ValueError("Invalid walkability result")
        return np.array([row["walkable"] for row in rows], dtype=bool)

    def paths(self, pairs):
        pairs = [{key: np.asarray(pair[key], dtype=float)[:2].tolist()
                  for key in ("start", "end")} for pair in pairs]
        # Full-precision endpoints must also fit the server's 16 KiB body limit.
        rows = self._query("paths_xy", "pairs", pairs, 64)
        for pair, row in zip(pairs, rows):
            if (not isinstance(row, dict)
                    or any(row.get(key) != pair[key] for key in ("start", "end"))):
                raise ValueError("Path response endpoints differ from request")
            path, length = row.get("path"), row.get("path_length")
            if path == [] and length is None:
                continue
            points = np.asarray(path, dtype=float)
            if (type(length) not in (float, int) or not math.isfinite(length) or length < 0
                    or points.ndim != 2 or points.shape[1] != 2 or not len(points)
                    or not np.isfinite(points).all()
                    or points[0].tolist() != pair["start"] or points[-1].tolist() != pair["end"]
                    or not math.isclose(float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()),
                                        length, rel_tol=1e-9, abs_tol=1e-9)):
                raise ValueError("Invalid freemap path geometry")
        return rows

    def path_exists(self, start, end):
        return self.paths([{"start": start, "end": end}])[0]["path_length"] is not None

    def node_distances(self, start, ends):
        rows = self.paths([{"start": start, "end": end} for end in ends])
        return [math.inf if row["path_length"] is None else float(row["path_length"]) for row in rows]

    def distances_from_agent(self, ends):
        return self.node_distances(self.camera_position, ends)

    def distance_from_agent(self, end):
        return self.distances_from_agent([end])[0]


def camera_inputs(frame):
    if frame.get("depth_convention") != "optical_z" or frame.get("depth_unit") != "m":
        raise ValueError("Mapper requires optical-Z depth in metres")
    intrinsic = np.array(frame["intrinsics"], dtype=float, copy=True)
    transform = np.asarray(frame["camera_to_map"], dtype=float)
    shape = tuple(frame["size_hw"])
    if (intrinsic.shape != (3, 3) or transform.shape != (4, 4)
            or not np.isfinite(intrinsic).all() or not np.isfinite(transform).all()
            or len(shape) != 2 or any(type(value) is not int or value <= 0 for value in shape)
            or intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0
            or not np.array_equal(intrinsic[2], [0, 0, 1])
            or intrinsic[0, 1] != 0 or intrinsic[1, 0] != 0
            or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-6)
            or not math.isclose(np.linalg.det(transform[:3, :3]), 1., abs_tol=1e-6)):
        raise ValueError("Invalid RGB-D camera calibration")
    # 原 Mapper 相机轴为右/上/后；R_map←mapper = R_map←optical · diag(1, -1, -1)。
    return intrinsic, transform[:3, 3].copy(), transform[:3, :3] @ np.diag([1., -1., -1.]), shape


class IsaacMapper:
    def __init__(self, packet, client, *, perceiver, save_dir, device,
                 floor_world_z, ceiling_world_z):
        from mapper_with_process_obs import Instruct_Mapper

        intrinsic, position, rotation, shape = camera_inputs(packet["frame"])
        if not (math.isfinite(floor_world_z) and math.isfinite(ceiling_world_z)
                and floor_world_z < ceiling_world_z):
            raise ValueError("Explicit finite world-height bounds are required")
        self.queries = IsaacMapperQueries(client, position)
        self.intrinsic, self.shape = intrinsic, shape
        self.mapper = Instruct_Mapper(intrinsic, perceiver=perceiver, save_dir=str(save_dir),
            device=device, floor_height=floor_world_z - position[2],
            ceiling_height=ceiling_world_z - position[2], translation_func=np.asarray,
            rotation_func=np.asarray, geometry_queries=self.queries, image_shape=shape)
        self.mapper.initialize(position, rotation, env=None)
        self._step = 0
        self._clouds, self._angles = [], []
        self._arrival_node = None
        self._distances = {}
        self._image_ref = None
        self.candidates = []

    def update(self, packet, *, episode_idx, episode_step, on_node_flag, current_node_idx):
        intrinsic, position, rotation, shape = camera_inputs(packet["frame"])
        if not np.array_equal(intrinsic, self.intrinsic) or shape != self.shape:
            raise ValueError("Camera calibration changed during mapping")
        root, frame = Path(packet["capture_root"]), packet["frame"]
        with Image.open(root / frame["rgb_path"]) as image:
            rgb = np.array(image.convert("RGB"))
        depth = np.load(root / frame["depth_path"], allow_pickle=False)
        if depth.shape != shape or rgb.shape != (*shape, 3):
            raise ValueError("RGB-D dimensions disagree with calibration")
        depth = np.where(np.isfinite(depth) & (depth > 0), depth, 0)
        self.queries.camera_position = position.copy()
        self.mapper.update(rgb, depth, position, rotation, episode_idx, episode_step,
                           on_node_flag, current_node_idx)

    def begin_observation(self):
        self._clouds, self._angles = [], []
        self.mapper.current_obj_indices = []

    def add(self, packet):
        from mapping_utils.geometry import gpu_merge_pointcloud

        self._step += 1
        self.update(packet, episode_idx=0, episode_step=self._step, on_node_flag=False,
                    current_node_idx=self.mapper.current_node_idx)
        self._distances, self.candidates = {}, []
        self._image_ref = str((Path(packet['capture_root']) / packet['frame']['rgb_path']).resolve())
        if self.mapper.current_pcd is not None and self.mapper.current_navigable_pcd is not None:
            cloud = gpu_merge_pointcloud(self.mapper.current_pcd, self.mapper.current_navigable_pcd)
            self._clouds.append(cloud.voxel_down_sample(self.mapper.pcd_resolution))
            self._angles.append(float(packet['pose']['yaw']))

    def finish_observation(self):
        mapper = self.mapper
        if not self._clouds or mapper.process_nav_pcd.is_empty() or mapper.traversable_pcd_all.is_empty():
            self._clouds, self._angles = [], []
            return False
        mapper.get_nodes(self._clouds, np.asarray(self._angles), self._arrival_node, 0, self._step)
        mapper.update_obj(mapper.current_node_idx, mapper.current_obj_indices)
        self._clouds, self._angles = [], []
        self._arrival_node = None
        return True

    def arrive(self, node_index):
        self._arrival_node = self.mapper.nodes[node_index]
        self.mapper.current_node_idx = node_index
        self.mapper.traj.append(node_index)

    def mark_observed(self, node_index):
        self.mapper.change_state(self.mapper.nodes[node_index])
        self._update_candidates()

    def _world_position(self, node):
        position = np.asarray(node.position, dtype=float) + self.mapper.initial_position
        position[2] = self.queries.camera_position[2]
        return tuple(float(value) for value in position)

    def _world_node(self, node):
        return SimpleNamespace(idx=node.idx, position=self._world_position(node), state=node.state,
            room_idx=node.room_idx, has_frontier=node.has_frontier, frontier_idxs=node.frontier_idxs)

    @property
    def nodes(self):
        return {node.idx: self._world_node(node) for node in self.mapper.nodes}

    def refresh(self, client, pose):
        position = np.asarray(pose['position'], dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError('Expected finite camera position')
        self.queries.camera_position = position.copy()
        self._distances, self.candidates = {}, []
        nodes = self.mapper.nodes
        distances = self.queries.distances_from_agent([self._world_position(node) for node in nodes])
        self._distances = {node.idx: distance for node, distance in zip(nodes, distances)
                           if math.isfinite(distance)}
        self._update_candidates()
        return list(self.candidates)

    def _update_candidates(self):
        self.candidates = [{'node_idx': node.idx, 'frontier_uid': f'node:{node.idx}',
            'position': list(self._world_position(node)), 'raw_distance_m': self._distances[node.idx],
            'frontier_idxs': node.frontier_idxs.tolist(), 'room_id': f'room:{node.room_idx}'}
            for node in self.mapper.nodes if node.state == 0 and node.idx in self._distances]

    def select(self, choose_room=None):
        from planning.exploration_policy import select_room_first
        from planning.room_policy import select_nearest_frontier_room

        mapper = self.mapper
        if not mapper.room_nodes:
            return None, 'no_observed_room', list(self.candidates)
        current = mapper.nodes[mapper.current_node_idx]
        room = mapper.room_nodes[current.room_idx]

        def room_choice():
            if choose_room is None:
                return select_nearest_frontier_room(mapper).room
            available = [SimpleNamespace(room_idx=index, state=item.state)
                         for index, item in enumerate(mapper.room_nodes)
                         if any(node.state == 0 and node.has_frontier and node.idx in self._distances
                                for node in item.nodes)]
            if not available:
                return None
            selected = choose_room(available)
            if selected is None:
                return None
            if not any(selected is item for item in available):
                raise ValueError('Room choice was not a supplied candidate')
            return mapper.room_nodes[selected.room_idx]

        node, source = select_room_first(current_room=room, rooms=mapper.room_nodes,
            local_choice=mapper.explore_in_room, remaining_choice=mapper.explore_after_fully_explored,
            room_choice=room_choice, closest_in_room=mapper.find_closet_viewpoint_in_room,
            semantic=choose_room is not None)
        self._update_candidates()
        if node is None or node.idx not in self._distances:
            return None, source, list(self.candidates)
        return self._world_node(node), source, list(self.candidates)

    def approach_targets(self, position):
        cloud = self.mapper.traversable_pcd
        if cloud.is_empty():
            return []
        points = cloud.point.positions.cpu().numpy() + self.mapper.initial_position
        order = np.argsort(np.linalg.norm(points[:, :2] - np.asarray(position)[:2], axis=1))[:32]
        points = points[order].copy()
        points[:, 2] = self.queries.camera_position[2]
        return points.tolist()

    def snapshot_fields(self, objects=()):
        from scene.contracts import FrontierSnapshot, Pose3D, RoomSnapshot, ViewpointSnapshot

        mapper = self.mapper
        room_ids = {index: f'room:{index}' for index in range(len(mapper.room_nodes))}
        published = {obj.uid for obj in objects}
        viewpoints, frontiers = [], []
        for node in mapper.nodes:
            position = self._world_position(node)
            metadata = {'source': 'original_instruct_mapper', 'frame_id': 'map', 'node_idx': node.idx,
                'geometry_domain': self.queries.domain, 'path_length': self._distances.get(node.idx),
                'reachable': node.idx in self._distances, 'frontier_idxs': node.frontier_idxs.tolist()}
            viewpoints.append(ViewpointSnapshot(uid=f'node:{node.idx}', pose=Pose3D(position),
                room_id=room_ids.get(node.room_idx), explored=bool(node.state), metadata=metadata))
            if node.has_frontier and node.state == 0:
                frontiers.append(FrontierSnapshot(uid=f'node:{node.idx}', position=position,
                    room_id=room_ids.get(node.room_idx), metadata=metadata))
        neighbors = {index: set() for index in room_ids}
        for first, second in mapper.get_edges():
            a, b = mapper.nodes[first].room_idx, mapper.nodes[second].room_idx
            if a in room_ids and b in room_ids and a != b:
                neighbors[a].add(room_ids[b])
                neighbors[b].add(room_ids[a])
        rooms = []
        for index, room in enumerate(mapper.room_nodes):
            positions = [node.position + mapper.initial_position for node in room.nodes]
            center = np.mean(positions, axis=0) if positions else None
            if center is not None:
                center[2] = mapper.floor_height + mapper.initial_position[2]
            uids = {mapper.objects[obj_index].uid for node in room.nodes for obj_index in node.objects}
            rooms.append(RoomSnapshot(uid=room_ids[index], label=None,
                centroid=None if center is None else tuple(float(value) for value in center),
                neighbors=tuple(sorted(neighbors[index])), objects=tuple(sorted(uids & published)),
                frontiers=tuple(item.uid for item in frontiers if item.room_id == room_ids[index]),
                explored=bool(room.state), metadata={'source': 'original_instruct_mapper',
                    'room_idx': index, 'node_ids': [node.idx for node in room.nodes], 'semantic_gt_used': False}))
        return {'rooms': tuple(rooms), 'viewpoints': tuple(viewpoints), 'frontiers': tuple(frontiers)}
