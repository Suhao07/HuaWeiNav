"""Goal-approach helpers for VLN target verification.

This module handles geometric movement around an already selected target
candidate. It does not run the final instruction verifier; the agent remains
the owner of semantic accept/reject decisions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger

from mapping_utils.transform import habitat_rotation
from navigation.action_controller import habitat_waypoint, next_action_to_waypoint
from navigation_core.view_geometry import camera_pitch, pitch_rotation
from planning.viewpoint_policy import score_object_view

def distance_to_object(agent: Any) -> float:
    """Compute the closest 2D distance from the robot to the current object."""

    positions = agent.object_final.pcd.point.positions.cpu().numpy()
    return float(np.min(np.linalg.norm(positions[:, :2] - agent.mapper.current_position[:2], axis=1)))


def tilt_actions(agent: Any) -> dict[str, tuple[int, float]]:
    """Return the platform's declared camera-tilt actions.

    俯仰能力以平台动作空间的声明为准：Habitat v1 声明 look_up/look_down，
    v0 与固定相机的轮式平台不声明。

    Args:
        agent: Habitat agent bound to the episode simulator.
    Returns:
        ``{"look_up"|"look_down": (action id, step in radians)}``; empty when the
        camera pitch is fixed.
    """

    action_space = agent.env.sim.sim_config.agents[0].action_space
    return {spec.name: (action, float(np.radians(spec.actuation.amount)))
            for action, spec in action_space.items() if spec.name in ("look_up", "look_down")}


def recheck_pitches(agent: Any) -> tuple[float, ...]:
    """Return executable recheck pitches.

    Args:
        agent: Habitat agent bound to the episode simulator.
    Returns:
        ``(0, -theta_tilt)`` in radians when LOOK_DOWN is declared; otherwise the
        fixed mounted pitch alone.
    """

    tilt = tilt_actions(agent)
    if "look_down" not in tilt:
        return (camera_pitch(np.asarray(agent.mapper.current_rotation)),)
    return 0.0, -tilt["look_down"][1]


def tilt_camera(agent: Any, pitch: float) -> bool:
    r"""Step the declared tilt action until the camera reaches the target pitch.

    .. math::
        n = \operatorname{round}\big(|\phi - \phi_0| / \theta_{tilt}\big)

    其中 \phi_0 为当前光轴仰角；\phi>\phi_0 执行 look_up，否则 look_down。
    未声明俯仰动作的平台保持安装俯仰，不执行动作。

    Args:
        agent: Habitat agent.
        pitch: Target pitch in radians; 0 is level.
    Returns:
        False if the episode ended while tilting.
    """

    tilt = tilt_actions(agent)
    if not tilt:
        return True
    current = camera_pitch(np.asarray(agent.mapper.current_rotation))
    action, step = tilt["look_up" if pitch > current else "look_down"]
    for _ in range(int(np.round(abs(pitch - current) / step))):
        agent.obs = agent.execution.step(action)
        agent.update_trajectory(agent.on_node_flag)
        if agent.env.episode_over:
            return False
    return True


def rotate_toward_object_for_recheck(agent: Any) -> bool:
    r"""Face the selected object in yaw and pitch before bbox re-verification.

    偏航按离散转向对准目标中心；俯仰在可执行角度中按执行位姿的实际投影
    选择，与视点规划同一判据：

    .. math::
        \phi^* = \arg\max_{\phi} \big(\mathbb{1}[d(x,R_\phi) \le d_{stop}],\ s(x,R_\phi)\big),\qquad
        R_\phi = R\,R_x(\phi - \phi_0)

    其中 x、R 为转向后的相机位姿，\phi_0 为当前仰角。

    Args:
        agent: Habitat agent with the selected ``object_final``.
    Returns:
        False if the episode ended while turning or tilting.
    """

    final_pos = agent.object_final.position[:2]
    nw_pos = agent.mapper.current_position[:2]
    nw_ori = habitat_rotation(agent.rotation)
    nw_ori = np.array([-nw_ori[0, 2], -nw_ori[1, 2]])
    final_pos = final_pos - nw_pos

    nw_ori = nw_ori / np.linalg.norm(nw_ori)
    final_pos = final_pos / np.linalg.norm(final_pos)
    dot_product = np.clip(np.dot(nw_ori, final_pos), -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(dot_product))
    direction = 3 if np.cross(nw_ori, final_pos) > 0 else 2
    num_turns = int(np.round(angle_deg / 30))

    logger.info("Now step: {}", agent.episode_steps)
    logger.info("Direction: {}", direction)
    logger.info("Num turns: {}", num_turns)

    # 复核前只做最小必要转向，避免为了确认目标额外触发规划状态变化。
    for _ in range(num_turns):
        agent.obs = agent.execution.step(direction)
        agent.update_trajectory(agent.on_node_flag)
        if agent.env.episode_over:
            return False

    rotation = np.asarray(agent.mapper.current_rotation)
    current_pitch = camera_pitch(rotation)
    height, width = agent.rgb_trajectory[-1].shape[:2]
    scored = []
    for pitch in recheck_pitches(agent):
        view = score_object_view(
            object_node=agent.object_final, camera_intrinsic=agent.mapper.camera_intrinsic,
            position=np.asarray(agent.mapper.current_position, dtype=float),
            rotation=rotation @ pitch_rotation(pitch - current_pitch),
            success_distance=agent._augment_final_stop_geometry({})["required_stop_distance"],
            image_width=width, image_height=height)
        if view is not None:
            scored.append((view[1], view[0].score, pitch))
    pitch = max(scored)[2] if scored else current_pitch
    logger.info("Recheck pitch: {:.1f} deg", np.degrees(pitch))
    return tilt_camera(agent, pitch)


def action_after_instruction_reject(agent: Any) -> int:
    """Recover from a failed final-instruction check and return the next action."""

    _ = habitat_waypoint(agent)
    continue_better_view = getattr(agent, "_continue_better_view_control", None)
    if callable(continue_better_view) and continue_better_view():
        return next_action_to_waypoint(agent)
    if agent.need_check_again:
        act = next_action_to_waypoint(agent)
        if act == 0:
            agent.need_check_again = False
            if not (callable(continue_better_view) and continue_better_view()):
                agent.after_check_again()
            _ = habitat_waypoint(agent)
    else:
        agent.after_check_again()
        _ = habitat_waypoint(agent)
    return next_action_to_waypoint(agent)
