"""Reproduce path_generator.m and verify its existing collision lookup table.

Run with NumPy and SciPy installed. Generation is offline; the robot loads the
versioned PLY files without either numerical dependency at runtime.
"""

from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree


def main():
    """Generate the three PLY assets after checking all grid/path associations.

    Raises:
        ValueError: Generated geometry disagrees with the existing lookup table.
        AssertionError: The existing lookup table has invalid dimensions or IDs.
    """
    folder = Path(__file__).resolve().parents[1] / "real_robot/ros2_ws/src/local_planner/paths"
    starts, paths, ends = [], [], []
    start_r = np.arange(101) * 0.01
    radius = np.arange(301) * 0.01
    knots = np.r_[start_r, 2.0, 2.999, 3.0]
    for group, shift1 in enumerate(np.arange(-3, 4) * 9.0):
        # MATLAB spline 的两个节点退化为线性插值；后续多节点使用 not-a-knot 三次样条。
        start_shift = start_r * shift1
        xy = np.column_stack((start_r * np.cos(np.deg2rad(start_shift)),
                              start_r * np.sin(np.deg2rad(start_shift))))
        starts.append(np.column_stack((xy, np.zeros(101), np.full(101, group))))
        for shift2 in shift1 + np.arange(-3, 4) * 9.0 * 0.65:
            for shift3 in shift2 + np.arange(-3, 4) * 9.0 * 0.65**2:
                angles = CubicSpline(knots, np.r_[start_shift, shift2, shift3, shift3])(radius)
                # 极坐标模型：x(r)=r*cos(theta(r)), y(r)=r*sin(theta(r))；角度转为弧度。
                xy = np.column_stack((radius * np.cos(np.deg2rad(angles)),
                                      radius * np.sin(np.deg2rad(angles))))
                path_id = len(paths)
                path = np.column_stack((xy, np.zeros(301), np.full(301, path_id), np.full(301, group)))
                paths.append(path)
                ends.append(path[-1])

    # 与 MATLAB rangesearch 等价：栅格中心到轨迹任一采样点距离 <= 0.45 m 即关联。
    # 逐路径查询最近点，避免一次构造全部邻接点列表占用大量内存。
    x = 3.2 - 0.02 * np.arange(161)
    scale_y = x / 3.2 + 0.45 / 4.5 * (3.2 - x) / 3.2
    grid = np.column_stack((np.repeat(x, 451),
                            (scale_y[:, None] * (4.5 - 0.02 * np.arange(451))).ravel()))
    expected = np.column_stack([cKDTree(path[:, :2]).query(grid)[0] <= 0.45 for path in paths])
    observed = np.zeros_like(expected)
    rows = (folder / "correspondences.txt").read_text().splitlines()
    assert len(rows) == len(grid), "collision grid row count differs"
    for index, row in enumerate(rows):
        values = list(map(int, row.split()))
        assert values[0] == index and values[-1] == -1, f"invalid collision row {index}"
        ids = values[1:-1]
        assert all(0 <= value < 343 for value in ids), f"invalid path ID in row {index}"
        observed[index, ids] = True
    differences = int(np.count_nonzero(expected != observed))
    if differences:
        raise ValueError(f"Existing correspondences differ at {differences} grid/path pairs; no assets written")
    print(f"Verified {len(grid)} grid cells x {len(paths)} paths: all collision associations match")

    for name, data, properties, fmt in (
        ("startPaths.ply", np.vstack(starts), ["float x", "float y", "float z", "int group_id"], "%f %f %f %d"),
        ("paths.ply", np.vstack(paths), ["float x", "float y", "float z", "int path_id", "int group_id"], "%f %f %f %d %d"),
        ("pathList.ply", np.vstack(ends), ["float end_x", "float end_y", "float end_z", "int path_id", "int group_id"], "%f %f %f %d %d"),
    ):
        header = "ply\nformat ascii 1.0\nelement vertex " + str(len(data)) + "\n"
        header += "".join("property " + prop + "\n" for prop in properties) + "end_header"
        np.savetxt(folder / name, data, fmt=fmt, header=header, comments="")
        print(f"Wrote {name}: {len(data)} vertices")


if __name__ == "__main__":
    main()
