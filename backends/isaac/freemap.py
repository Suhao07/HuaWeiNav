"""Static freemap XY walkability shared by the simulator planner and the policy process."""
import numpy as np


EDGE_TOLERANCE = 1e-10


class FreemapWalkability:
    """Closed-cell supercover walkability on the official binary freemap.

    数学模型：设某一维的格中心升序排列为 c_0 < ... < c_{n-1}，格边界
    e_0 = c_0 - (c_1 - c_0)/2，e_i = (c_{i-1} + c_i)/2，e_n = c_{n-1} + (c_{n-1} - c_{n-2})/2，
    则格 i 占据闭区间 [e_i, e_{i+1}]。点 p = (x, y) 可行走当且仅当
        x ∈ [e^x_0, e^x_n] ∧ y ∈ [e^y_0, e^y_m] ∧
        ∀ 格 (r, c)：x ∈ [e^x_c - τ, e^x_{c+1} + τ] ∧ y ∈ [e^y_r - τ, e^y_{r+1} + τ] ⇒ grid[r, c] = 1，
    其中 τ = EDGE_TOLERANCE。落在格边界或角点上的点要求其触及的全部格自由，
    因此官方最近格判定 grid[argmin|c - x|] = 1 被该定义蕴含；τ 远小于格宽，
    每一维最多触及两格。栅格是场景静态数据，策略进程持有一份副本后即可本地查表。
    """

    def __init__(self, x_coords, y_coords, grid):
        """Validate and index the freemap geometry.

        Args:
            x_coords: 列中心坐标，严格单调（升序或降序）。
            y_coords: 行中心坐标，严格单调（升序或降序）。
            grid: 形状为 (len(y_coords), len(x_coords)) 的 0/1 栅格，1 为自由格。

        Raises:
            ValueError: 坐标少于两个、非有限、非严格单调，或栅格形状/取值不合法。
        """
        self.grid = np.asarray(grid)
        self.axes = [np.asarray(x_coords, dtype=float), np.asarray(y_coords, dtype=float)]
        self.edges, self.descending, spacings = [], [], []
        for axis in self.axes:
            if axis.ndim != 1 or len(axis) < 2 or not np.isfinite(axis).all():
                raise ValueError("freemap axes must have at least two finite coordinates")
            diffs = np.diff(axis)
            if not ((diffs > 0).all() or (diffs < 0).all()):
                raise ValueError("freemap axes must be strictly monotonic")
            self.descending.append(bool(diffs[0] < 0))
            centers = axis[::-1] if self.descending[-1] else axis
            self.edges.append(np.concatenate(([centers[0] - (centers[1] - centers[0]) / 2],
                                              (centers[:-1] + centers[1:]) / 2,
                                              [centers[-1] + (centers[-1] - centers[-2]) / 2])))
            spacings.append(np.abs(diffs).min())
        if self.grid.shape != (len(self.axes[1]), len(self.axes[0])):
            raise ValueError("freemap grid shape disagrees with coordinates")
        if not np.isin(self.grid, (0, 1)).all():
            raise ValueError("freemap grid must be binary")
        self.grid = self.grid.astype(np.uint8)
        self.min_spacing = float(min(spacings))

    def _touched(self, values, dimension):
        """Return (low, high) grid indices touched along one dimension, in grid order."""
        edges = self.edges[dimension]
        last = len(edges) - 2
        low = np.clip(np.searchsorted(edges, values - EDGE_TOLERANCE, side="left") - 1, 0, last)
        high = np.clip(np.searchsorted(edges, values + EDGE_TOLERANCE, side="right") - 1, 0, last)
        if self.descending[dimension]:
            return last - high, last - low
        return low, high

    def mask(self, points):
        """Vectorised walkability for an (N, 2) array of XY world coordinates.

        Args:
            points: 可转换为 (N, 2) 浮点数组的 XY 坐标；非有限坐标判为不可行走。

        Returns:
            形状 (N,) 的布尔数组。

        Raises:
            ValueError: 输入不是 (N, 2) 形状。
        """
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Expected an N by 2 point array")
        inside = np.ones(len(points), dtype=bool)
        for dimension, edges in enumerate(self.edges):
            inside &= (points[:, dimension] >= edges[0]) & (points[:, dimension] <= edges[-1])
        col_low, col_high = self._touched(points[:, 0], 0)
        row_low, row_high = self._touched(points[:, 1], 1)
        grid = self.grid
        free = (grid[row_low, col_low] & grid[row_low, col_high]
                & grid[row_high, col_low] & grid[row_high, col_high])
        return inside & (free == 1)

    def is_walkable(self, point):
        """Scalar form of mask for a single XY point."""
        return bool(self.mask(np.asarray([point], dtype=float))[0])
