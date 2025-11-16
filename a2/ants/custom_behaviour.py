import numpy as np
from typing import Optional
from irsim.lib import register_behavior
from numpy import ndarray

# --- Basic Parameters ---
MAX_SPEED = 0.3
ANGULAR_GAIN = 2.0

# Collision avoidance (LiDAR)
SEPARATION_WEIGHT = 1.3
DESIRED_DISTANCE = 0.75
LIDAR_OFFSET = 0.05

# Leader’s goal (kept for compatibility with main.py, but not used for motion)
GOAL_POS = {'x': 5, 'y': 5}

# --- Pheromone Parameters ---
PHERO_CELL_SIZE = 0.1
PHERO_DECAY = 0.015
PHERO_DIFFUSE = 0.7
PHERO_DEPOSIT = 10.0
PHERO_NOISE_THRESH = 0.001
PHERO_WORLD_MIN = (-20.0, -20.0)
PHERO_WORLD_MAX = (20.0, 20.0)


# -----------------------------------------------------
# Utility
# -----------------------------------------------------
def _to_floats(pos: ndarray) -> list[float]:
    return [float(p) for p in pos]


def _safe_unit(v: ndarray) -> ndarray:
    n = np.linalg.norm(v)
    return v / (n + 1e-6)


def _compute_lidar_points(pos: ndarray, th: float, lidar) -> ndarray:
    scan_data: dict = lidar.get_scan()
    ranges: ndarray = np.array(scan_data["ranges"])
    angles: ndarray = np.linspace(lidar.angle_min, lidar.angle_max, len(ranges))
    lidar_points = [
        [
            pos[0] + r * np.cos(ang + th),
            pos[1] + r * np.sin(ang + th)
        ]
        for r, ang in zip(ranges, angles)
        if LIDAR_OFFSET < r < lidar.range_max - LIDAR_OFFSET
    ]
    return np.array(lidar_points) if len(lidar_points) else np.zeros((0, 2))


# -----------------------------------------------------
# Pheromone Field (grid with diffusion + decay)
# -----------------------------------------------------
class PheromoneField:
    def __init__(self, world_min, world_max, cell_size):
        self.world_min = np.array(world_min, float)
        self.world_max = np.array(world_max, float)
        self.cell_size = float(cell_size)
        size = np.ceil((self.world_max - self.world_min) / self.cell_size).astype(int)
        self.ny, self.nx = int(size[1]), int(size[0])
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
            return iy, ix
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
        self.grid *= (1.0 - decay)
        # Diffuse
        if diffuse > 0:
            g = self.grid
            pad = np.pad(g, 1, mode='edge')
            new_g = (
                self.kernel[0, 0] * pad[0:-2, 0:-2] + self.kernel[0, 1] * pad[0:-2, 1:-1] + self.kernel[0, 2] * pad[0:-2, 2:] +
                self.kernel[1, 0] * pad[1:-1, 0:-2] + self.kernel[1, 1] * pad[1:-1, 1:-1] + self.kernel[1, 2] * pad[1:-1, 2:] +
                self.kernel[2, 0] * pad[2:,   0:-2] + self.kernel[2, 1] * pad[2:,   1:-1] + self.kernel[2, 2] * pad[2:,   2:]
            )
            self.grid = (1 - diffuse) * g + diffuse * new_g

    def gradient(self, pos: ndarray) -> np.ndarray:
        """Return continuous pheromone gradient vector at world pos."""
        idx = self._to_idx(pos)
        if idx is None:
            return np.zeros(2)
        iy, ix = idx
        # Clamp neighborhood indices
        ix0, ix1 = max(1, ix - 1), min(self.nx - 2, ix + 1)
        iy0, iy1 = max(1, iy - 1), min(self.ny - 2, iy + 1)
        # Finite difference gradient
        gx = (self.grid[iy, ix1] - self.grid[iy, ix0]) / (2 * self.cell_size)
        gy = (self.grid[iy1, ix] - self.grid[iy0, ix]) / (2 * self.cell_size)
        return np.array([gx, gy], dtype=float)


# Global pheromone field
_PHERO = PheromoneField(PHERO_WORLD_MIN, PHERO_WORLD_MAX, PHERO_CELL_SIZE)

# -----------------------------------------------------
# Leader random-walk + trajectories
# -----------------------------------------------------
_LEADER_DIR = np.array([1.0, 0.0], dtype=float)

# Leader trajectory: list of [x, y]
_LEADER_TRAJECTORY: list[list[float]] = []

# Follower name → list of [x, y] positions per step
_FOLLOWER_TRAJECTORIES: dict[str, list[list[float]]] = {}


def _separation_from_lidar(pos: ndarray, lidar_points: ndarray) -> ndarray:
    sep = np.zeros(2)
    for npos in lidar_points:
        diff = pos - npos
        d = np.linalg.norm(diff)
        if 1e-6 < d < DESIRED_DISTANCE:
            sep += diff / (d ** 2)
    return sep


def _update_leader_random_direction() -> ndarray:
    """Smooth random-walk direction for the leader in world frame."""
    global _LEADER_DIR
    dtheta = np.random.uniform(-0.3, 0.3)  # radians per step
    c, s = np.cos(dtheta), np.sin(dtheta)
    R = np.array([[c, -s],
                  [s,  c]], dtype=float)
    _LEADER_DIR = _safe_unit(R @ _LEADER_DIR)
    return _LEADER_DIR

def reset_efficiency():
    """
    Reset stored trajectories for leader and followers.
    Call this before starting each new simulation run.
    """
    _LEADER_TRAJECTORY.clear()
    _FOLLOWER_TRAJECTORIES.clear()

    global _LEADER_DIR
    _LEADER_DIR = np.array([1.0, 0.0], dtype=float)



def print_final_efficiency():
    """
    Compute average distance between each follower and the leader over the run.

    - Leader logs its own position in _LEADER_TRAJECTORY.
    - Each follower logs only its own ego position in _FOLLOWER_TRAJECTORIES.
    - Steps are aligned by index: index k is timestep k.
    - We use steps up to the minimum length over leader + followers.
    """
    if not _LEADER_TRAJECTORY:
        print("[EFF] No leader data recorded.")
        return None

    if not _FOLLOWER_TRAJECTORIES:
        print("[EFF] No follower data recorded.")
        return None

    leader_arr = np.array(_LEADER_TRAJECTORY, dtype=float)

    # Only followers that actually moved
    followers = [
        (name, np.array(traj, float))
        for name, traj in _FOLLOWER_TRAJECTORIES.items()
        if len(traj) > 1
    ]

    if not followers:
        print("[EFF] Followers did not move enough to compute distances.")
        return None

    # Timestep count limited by the shortest trajectory among leader + followers
    min_steps = min(
        len(leader_arr),
        *(len(traj) for _, traj in followers)
    )

    if min_steps == 0:
        print("[EFF] No overlapping timesteps between leader and followers.")
        return None

    total_dist_sum = 0.0
    total_dist_count = 0

    for step in range(min_steps):
        leader_pos = leader_arr[step]
        for _, traj in followers:
            follower_pos = traj[step]
            d = np.linalg.norm(follower_pos - leader_pos)
            total_dist_sum += d
            total_dist_count += 1

    if total_dist_count == 0:
        print("[EFF] No leader–follower distances could be computed.")
        return None

    avg_dist = total_dist_sum / total_dist_count

    print("\n============================")
    print(" FINAL EFFICIENCY RESULTS")
    print(f" Followers:                    {len(followers)}")
    print(f" Steps used (aligned):         {min_steps}")
    print(f" Total leader–follower pairs:  {total_dist_count}")
    print(f" Avg distance from leader:     {avg_dist:.3f}")
    print("============================\n")

    return avg_dist


@register_behavior('diff', 'custom_behaviour')
def move(ego_object, objects=None, **kw) -> ndarray:
    x, y, th = _to_floats(ego_object.state)
    pos = np.array([x, y], dtype=float)
    lidar_points = _compute_lidar_points(pos, th, ego_object.lidar)

    # Leader deposits pheromone first (so followers can sense it immediately)
    if ego_object.name == 'robot_0':
        _PHERO.deposit(pos, PHERO_DEPOSIT)

    # Decay & diffuse field
    _PHERO.step(decay=PHERO_DECAY, diffuse=PHERO_DIFFUSE)

    # --- Leader: random wandering ---
    if ego_object.name == 'robot_0':
        direction = _update_leader_random_direction()
        linear_velocity = MAX_SPEED

        # Log leader position
        _LEADER_TRAJECTORY.append([x, y])

    else:
        # Log only this follower's own position (ego state)
        if ego_object.name not in _FOLLOWER_TRAJECTORIES:
            _FOLLOWER_TRAJECTORIES[ego_object.name] = []
        _FOLLOWER_TRAJECTORIES[ego_object.name].append([x, y])

        # --- Follower: move up pheromone gradient ---
        grad = _PHERO.gradient(pos)
        mag = np.linalg.norm(grad)
        if mag < PHERO_NOISE_THRESH:
            direction = np.zeros(2)
            linear_velocity = 0.0
        else:
            direction = _safe_unit(grad)
            linear_velocity = MAX_SPEED

    # --- LiDAR avoidance ---
    separation = _separation_from_lidar(pos, lidar_points)
    control_vec = _safe_unit(direction + SEPARATION_WEIGHT * separation)

    # Convert to angular + linear velocities
    desired_yaw = np.arctan2(control_vec[1], control_vec[0])
    angular_velocity = ANGULAR_GAIN * np.arctan2(
        np.sin(desired_yaw - th), np.cos(desired_yaw - th)
    )

    # Reduce speed if very close to obstacle
    if lidar_points.size > 0:
        dists = np.linalg.norm(lidar_points - pos, axis=1)
        if len(dists) > 0 and np.min(dists) < 0.4 * DESIRED_DISTANCE:
            linear_velocity = min(linear_velocity, MAX_SPEED * 0.2)

    return np.array(
        [[float(linear_velocity)], [float(np.clip(angular_velocity, -1.0, 1.0))]],
        dtype=float
    )
