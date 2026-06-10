"""Project object point clouds back into panoramic RGB evidence views.

The functions here translate a 3D object entity into the best available
panoramic image crop and projected bbox. They do not run detection, VLM label
refinement, or final instruction verification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from cv_utils.stitch import image_stitch_and_crop
from mapping_utils.geometry import project_to_camera
from mapping_utils.transform import habitat_rotation


@dataclass(frozen=True)
class ObjectViewEvidence:
    """RGB evidence and projected bbox for one object candidate."""

    image: np.ndarray
    bbox_xyxy: np.ndarray
    bbox_tensor: Any
    bbox_real_xyxy: np.ndarray | None


def _view_index_for_object(agent: Any, obj: Any) -> int:
    """Map object bearing in robot frame to the closest 15-degree panorama bin."""

    current_pos = agent.mapper.current_position[:2]
    nw_ori = habitat_rotation(agent.rotation)
    nw_ori = np.array([-nw_ori[0, 2], -nw_ori[1, 2]])
    obj.position = np.mean(obj.pcd.point.positions.cpu().numpy(), axis=0)
    center_pos = obj.position[:2] - current_pos

    nw_ori = nw_ori / np.linalg.norm(nw_ori)
    center_pos = center_pos / np.linalg.norm(center_pos)
    dot_product = np.clip(np.dot(nw_ori, center_pos), -1.0, 1.0)
    angle = np.degrees(np.arccos(dot_product))
    if np.cross(nw_ori, center_pos) < 0:
        angle = 360 - angle

    # 12 张原始视图之间还有 15 度半视角；奇数 bin 需要用相邻两帧拼接。
    angle += 7.5
    if angle >= 360:
        angle -= 360
    return int(angle / 15)


def _evidence_image_and_rotation(images: list[np.ndarray], rotations: list[Any], image_bin: int, camera_intrinsic: Any) -> tuple[np.ndarray, Any]:
    """Return the RGB evidence view and camera rotation for a 15-degree bin."""

    if image_bin % 2 == 0:
        image_index = int(image_bin / 2)
        return images[image_index], rotations[image_index]

    image_index = int(image_bin / 2)
    next_index = image_index + 1
    if next_index == len(images):
        next_index = 0
    image = image_stitch_and_crop(images[image_index], images[next_index], camera_intrinsic)
    rotation = rotations[image_index]
    deg = np.arccos(rotation[0][0])
    if rotation[0][2] > 0:
        deg = np.pi * 2 - deg
    deg += (15 / 180) * np.pi
    rotation = np.array([
        [np.cos(deg), 0, -np.sin(deg)],
        [np.sin(deg), 0, np.cos(deg)],
        [0, 1, 0],
    ])
    return image, rotation


def _project_bbox(obj_pcd: Any, camera_intrinsic: Any, position: Any, rotation: Any, *, padding: int = 3) -> np.ndarray | None:
    """Project an object point cloud and return a clamped image-space bbox."""

    camera_points = project_to_camera(obj_pcd, camera_intrinsic, position, rotation)
    camera_points = np.array(camera_points).T
    camera_points = np.array(camera_points[:, :2], dtype=np.int32)
    flag = (
        (camera_points[:, 0] >= 0) &
        (camera_points[:, 0] < 640) &
        (camera_points[:, 1] >= 0) &
        (camera_points[:, 1] < 480)
    )
    camera_points = camera_points[flag]
    if camera_points.shape[0] == 0:
        return None

    bbox = np.array([np.min(camera_points, axis=0), np.max(camera_points, axis=0)])
    bbox[0] = np.maximum(bbox[0] - padding, 0)
    bbox[1] = np.minimum(bbox[1] + padding, [639, 479])
    return np.array([bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]])


def project_object_view(agent: Any, obj: Any, images: list[np.ndarray], rotations: list[Any], torch_module: Any) -> ObjectViewEvidence | None:
    """Build RGB/bbox evidence for one merged object candidate."""

    image_bin = _view_index_for_object(agent, obj)
    image, rotation = _evidence_image_and_rotation(images, rotations, image_bin, agent.mapper.camera_intrinsic)
    bbox_xyxy = _project_bbox(
        obj.pcd_all,
        agent.mapper.camera_intrinsic,
        agent.mapper.current_position,
        rotation,
    )
    if bbox_xyxy is None:
        return None

    bbox_real = _project_bbox(
        obj.pcd,
        agent.mapper.camera_intrinsic,
        agent.mapper.current_position,
        rotation,
        padding=0,
    )
    return ObjectViewEvidence(
        image=image,
        bbox_xyxy=bbox_xyxy,
        bbox_tensor=torch_module.tensor(bbox_xyxy).unsqueeze(0),
        bbox_real_xyxy=bbox_real,
    )
