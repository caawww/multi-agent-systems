import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from irsim.lib import register_behavior
from numpy import ndarray

# --- Basic Parameters ---
MAX_SPEED = 0.3
ANGULAR_GAIN = 2.0

# Collision avoidance (LiDAR)
SEPARATION_WEIGHT = 1.3
DESIRED_DISTANCE = 0.75
STOPPING_DISTANCE = 1.0
LIDAR_OFFSET = 0.05

# Leader’s goal
GOAL_POS = {'x': 5, 'y': 5}

# --- Pheromone Parameters ---
PHERO_CELL_SIZE = 0.1
PHERO_DECAY = 0.015
PHERO_DIFFUSE = 0.7
PHERO_DEPOSIT = 10.0
PHERO_NOISE_THRESH = 0.001
# Match the simulation world shown in your plot (0..10)
PHERO_WORLD_MIN = (0.0, 0.0)
PHERO_WORLD_MAX = (10.0, 10.0)

# Visualization params
# update 5x less often than before
VIS_UPDATE_FREQ = 5   # update every N leader calls
_FIG = None
_AX = None
_IM = None
_vis_counter = 0

# -----------------------------------------------------
# Utility
# -----------------------------------------------------
def _to_floats(pos: ndarray) -> list[float]:
    return [float(p) for p in pos]


def _safe_unit(v: ndarray) -> ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        return np.zeros_like(v)
    return v / (n + 1e-6)


def _compute_lidar_points(pos: ndarray, th: float, lidar) -> ndarray:
    """
    Convert lidar scan into Nx2 array columns: x, y
    """
    if lidar is None:
        return np.zeros((0, 2))
    scan_data: dict = lidar.get_scan()
    ranges: ndarray = np.array(scan_data.get("ranges", []), dtype=float)
    if ranges.size == 0:
        return np.zeros((0, 2))
    angle_min = getattr(lidar, "angle_min", -0.5 * np.pi)
    angle_max = getattr(lidar, "angle_max", 0.5 * np.pi)
    angles: ndarray = np.linspace(angle_min, angle_max, len(ranges))
    pts = []
    for r, ang in zip(ranges, angles):
        if LIDAR_OFFSET < r < (getattr(lidar, "range_max", r) - LIDAR_OFFSET):
            x = pos[0] + r * np.cos(ang + th)
            y = pos[1] + r * np.sin(ang + th)
            pts.append([x, y])
    return np.array(pts) if len(pts) else np.zeros((0, 2))


# -----------------------------------------------------
# Pheromone Field (grid with diffusion + decay)
# -----------------------------------------------------
class PheromoneField:
    def __init__(self, world_min, world_max, cell_size):
        self.world_min = np.array(world_min, float)
        self.world_max = np.array(world_max, float)
        self.cell_size = float(cell_size)
        size = np.ceil((self.world_max - self.world_min) / self.cell_size).astype(int)
        size[size < 1] = 1
        self.nx, self.ny = int(size[0]), int(size[1])
        # store grid as [ny, nx] (rows=y, cols=x)
        self.grid = np.zeros((self.ny, self.nx), dtype=np.float32)
        self.kernel = np.array([[1, 2, 1],
                                [2, 4, 2],
                                [1, 2, 1]], dtype=np.float32)
        self.kernel /= self.kernel.sum()

    def _to_idx(self, pos: ndarray) -> Optional[tuple[int, int]]:
        rel = (pos - self.world_min) / self.cell_size
        ix = int(np.floor(rel[0]))
        iy = int(np.floor(rel[1]))
        if 0 <= ix < self.nx and 0 <= iy < self.ny:
            return iy, ix  # (row, col)
        return None

    def deposit(self, pos: ndarray, amount: float):
        idx = self._to_idx(pos)
        if idx is None:
            return
        iy, ix = idx
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                jy, jx = iy + dy, ix + dx
                if 0 <= jy < self.ny and 0 <= jx < self.nx:
                    self.grid[jy, jx] += amount * self.kernel[dy + 1, dx + 1]

    def step(self, decay: float, diffuse: float):
        # Decay
        self.grid *= (1.0 - float(decay))
        # Diffuse (3x3 kernel convolution via padded slicing)
        if diffuse > 0:
            g = self.grid
            pad = np.pad(g, 1, mode='edge')
            new_g = (
                self.kernel[0, 0] * pad[0:-2, 0:-2] + self.kernel[0, 1] * pad[0:-2, 1:-1] + self.kernel[0, 2] * pad[0:-2, 2:] +
                self.kernel[1, 0] * pad[1:-1, 0:-2] + self.kernel[1, 1] * pad[1:-1, 1:-1] + self.kernel[1, 2] * pad[1:-1, 2:] +
                self.kernel[2, 0] * pad[2:  , 0:-2] + self.kernel[2, 1] * pad[2:  , 1:-1] + self.kernel[2, 2] * pad[2:  , 2:]
            )
            self.grid = (1.0 - float(diffuse)) * g + float(diffuse) * new_g

    def gradient(self, pos: ndarray) -> np.ndarray:
        idx = self._to_idx(pos)
        if idx is None:
            return np.zeros(2)
        iy, ix = idx
        ix0, ix1 = max(1, ix - 1), min(self.nx - 2, ix + 1)
        iy0, iy1 = max(1, iy - 1), min(self.ny - 2, iy + 1)
        gx = (self.grid[iy, ix1] - self.grid[iy, ix0]) / (2 * self.cell_size)
        gy = (self.grid[iy1, ix] - self.grid[iy0, ix]) / (2 * self.cell_size)
        return np.array([gx, gy], dtype=float)

    def render(self, cmap='hot', vmin=None, vmax=None):
        """Real-time heatmap of pheromone grid (non-blocking)."""
        global _FIG, _AX, _IM
        extent = (float(self.world_min[0]), float(self.world_max[0]),
                float(self.world_min[1]), float(self.world_max[1]))

        # auto-scale vmax if not provided (use 99th percentile of nonzero values)
        if vmax is None:
            nz = self.grid[self.grid > 0]
            if nz.size > 0:
                vmax = float(max(np.percentile(nz, 99), np.max(nz)))
            else:
                vmax = 1.0

        if _FIG is None or _AX is None or _IM is None:
            plt.ion()
            _FIG, _AX = plt.subplots(figsize=(6, 6))
            # do NOT flip the grid; use origin='lower' so y increases upward
            _IM = _AX.imshow(self.grid, origin='lower', extent=extent,
                            cmap=cmap, interpolation='bilinear', vmin=vmin, vmax=vmax)
            _AX.set_title('Pheromone field')
            _AX.set_xlabel('x')
            _AX.set_ylabel('y')
            _FIG.colorbar(_IM, ax=_AX).set_label('pheromone')
            plt.show(block=False)
            _FIG.canvas.draw()
        else:
            _IM.set_data(self.grid)
            _IM.set_clim(vmin if vmin is not None else 0.0, vmax)
            _FIG.canvas.draw_idle()
        plt.pause(0.001)

# Global pheromone field (use world bounds that match your sim)
_PHERO = PheromoneField(PHERO_WORLD_MIN, PHERO_WORLD_MAX, PHERO_CELL_SIZE)


# -----------------------------------------------------
# Control Logic
# -----------------------------------------------------
def _separation_from_lidar(pos: ndarray, lidar_points: ndarray) -> ndarray:
    sep = np.zeros(2)
    for npos in lidar_points:
        diff = pos - npos
        d = np.linalg.norm(diff)
        if 1e-6 < d < DESIRED_DISTANCE:
            sep += diff / (d ** 2)
    return sep


@register_behavior('diff', 'custom_behaviour')
def move(ego_object, objects=None, **kw) -> ndarray:
    global _vis_counter
    x, y, th = _to_floats(ego_object.state)
    pos = np.array([x, y], dtype=float)
    lidar_points = _compute_lidar_points(pos, th, getattr(ego_object, 'lidar', None))

    # Leader deposits pheromone first
    if ego_object.name == 'robot_0':
        _PHERO.deposit(pos, PHERO_DEPOSIT)

    # Decay & diffuse field
    _PHERO.step(decay=PHERO_DECAY, diffuse=PHERO_DIFFUSE)

    # Movement logic
    if ego_object.name == 'robot_0':
        goal = np.array([GOAL_POS['x'], GOAL_POS['y']], dtype=float)
        to_goal = goal - pos
        dist = np.linalg.norm(to_goal)
        direction = _safe_unit(to_goal)
        linear_velocity = MAX_SPEED * max(0.0, min(1.0, dist - STOPPING_DISTANCE))
    else:
        grad = _PHERO.gradient(pos)
        mag = np.linalg.norm(grad)
        if mag < PHERO_NOISE_THRESH:
            direction = np.zeros(2)
            linear_velocity = 0.0
        else:
            direction = _safe_unit(grad)
            linear_velocity = MAX_SPEED

    separation = _separation_from_lidar(pos, lidar_points)
    control_vec = _safe_unit(direction + SEPARATION_WEIGHT * separation)

    desired_yaw = np.arctan2(control_vec[1], control_vec[0])
    angular_velocity = ANGULAR_GAIN * np.arctan2(
        np.sin(desired_yaw - th), np.cos(desired_yaw - th)
    )

    if lidar_points.size > 0:
        dists = np.linalg.norm(lidar_points - pos, axis=1)
        if dists.size > 0 and np.min(dists) < 0.4 * DESIRED_DISTANCE:
            linear_velocity = min(linear_velocity, MAX_SPEED * 0.2)

    # Render pheromone field occasionally (leader triggers updates)
    _vis_counter += 1
    if ego_object.name == 'robot_0' and (_vis_counter % VIS_UPDATE_FREQ) == 0:
        vmax = float(np.max(_PHERO.grid)) if np.any(_PHERO.grid) else 1.0
        _PHERO.render(cmap='hot', vmin=0.0, vmax=vmax)

    return np.array(
        [[float(linear_velocity)], [float(np.clip(angular_velocity, -1.0, 1.0))]],
        dtype=float
    )