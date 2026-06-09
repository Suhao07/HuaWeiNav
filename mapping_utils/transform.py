import numpy as np
import quaternion


def habitat_camera_intrinsic(config):
    # 兼容新版 cfg.habitat.* 和旧版 cfg.SIMULATOR.*，输出 RGB-D 反投影使用的 K 矩阵。
    if hasattr(config, "habitat"):
        sensors = config.habitat.simulator.agents.main_agent.sim_sensors
        camera_config = sensors.depth_sensor
        rgb_config = sensors.rgb_sensor
        width = camera_config.width
        height = camera_config.height
        hfov = camera_config.hfov
        rgb_width = rgb_config.width
        rgb_height = rgb_config.height
        rgb_hfov = rgb_config.hfov
    else:
        camera_config = config.SIMULATOR.DEPTH_SENSOR
        rgb_config = config.SIMULATOR.RGB_SENSOR
        width = camera_config.WIDTH
        height = camera_config.HEIGHT
        hfov = camera_config.HFOV
        rgb_width = rgb_config.WIDTH
        rgb_height = rgb_config.HEIGHT
        rgb_hfov = rgb_config.HFOV

    assert width == rgb_width, 'The configuration of the depth camera should be the same as rgb camera.'
    assert height == rgb_height, 'The configuration of the depth camera should be the same as rgb camera.'
    assert hfov == rgb_hfov, 'The configuration of the depth camera should be the same as rgb camera.'
    xc = (width - 1.) / 2.
    zc = (height - 1.) / 2.
    # pinhole camera: f = W / (2 * tan(hfov / 2))。
    f = (width / 2.) / np.tan(np.deg2rad(hfov / 2.))
    intrinsic_matrix = np.array([[f, 0, xc], [0, f, zc], [0, 0, 1]], np.float32)
    return intrinsic_matrix


def habitat_translation(position):
    return np.array([position[0], position[2], position[1]])


def habitat_rotation(rotation):
    rotation_matrix = quaternion.as_rotation_matrix(rotation)
    transform_matrix = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    rotation_matrix = np.matmul(transform_matrix, rotation_matrix)
    return rotation_matrix
