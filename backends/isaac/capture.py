"""Export Isaac observations using STRIVE's shared optical-camera protocol."""
import numpy as np

from prior_map.observations import write_frame


def capture_frame(env, root, frame_id):
    """Persist synchronized RGB, optical-Z depth and actual USD camera pose.

    USD camera axes are right/up/backward; the shared protocol is right/down/
    forward. Thus T_map_optical = T_map_usd @ diag(1, -1, -1, 1).
    Read the actual transform rather than reconstructing it from agent yaw.
    """
    from pxr import UsdGeom

    rgb = env.render_rgb()
    depth = env.render_depth()
    if depth is None:
        raise RuntimeError("Isaac depth annotator produced no frame")
    invalid = ~np.isfinite(depth) | (depth <= 0)
    valid_fraction = float((~invalid).mean())
    if valid_fraction == 0:
        raise RuntimeError("Isaac depth frame contains no valid measurements")
    depth = np.where(invalid, 0, depth).astype(np.float32)
    camera = UsdGeom.Camera(env._app.context.get_stage().GetPrimAtPath(env._camera_prim_path))
    # Gf matrices store row-vector transforms; portable records use columns.
    pose = np.asarray(UsdGeom.Xformable(camera).ComputeLocalToWorldTransform(0)).T
    pose = pose @ np.diag([1, -1, -1, 1])
    height, width = depth.shape
    focal = camera.GetFocalLengthAttr().Get()
    intrinsic = np.array([[width * focal / camera.GetHorizontalApertureAttr().Get(), 0, width / 2],
                          [0, height * focal / camera.GetVerticalApertureAttr().Get(), height / 2],
                          [0, 0, 1]])
    return write_frame(root, frame_id, rgb, depth, intrinsic, pose,
                       valid_depth_fraction=valid_fraction)
