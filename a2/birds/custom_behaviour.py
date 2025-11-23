import math
from typing import Optional

import numpy as np
from irsim.lib import register_behavior
from numpy import ndarray
from scipy.spatial.distance import pdist, squareform

# --- Parameters ---
MAX_SPEED: float = 0.3
ANGULAR_GAIN: float = 2.0
COHESION_WEIGHT: float = 0.6
SEPARATION_WEIGHT: float = 1.2
ALIGNMENT_WEIGHT: float = 0.2
DESIRED_DISTANCE: float = 0.75
STOPPING_DISTANCE: float = 1.0
LIDAR_OFFSET: float = 0.05
GOAL_POS: dict[str, float] = {'x': 5, 'y': 10}

INTERACTIVE = True
FOLLOWER_DIST = list()
LEADER_NAME = 'robot_0'
_ROBOT_POS = {}


def _to_floats(pos: ndarray) -> list[float]:
    """Convert pos list to plain floats."""
    return [float(p) for p in pos]


def _boid_control(pos: np.ndarray, lidar_points: np.ndarray, leader_pos: Optional[np.ndarray] = None) -> np.ndarray:
    """Boid control using LiDAR position and velocity data."""

    cohesion = np.zeros(2)
    separation = np.zeros(2)
    alignment = np.zeros(2)

    if lidar_points.size > 0:
        positions = lidar_points[:, :2]  # (x, y)
        velocities = lidar_points[:, 2:]  # (vx, vy)

        # --- Cohesion ---
        if leader_pos is not None:
            vec_to_leader = leader_pos - pos
            cohesion = vec_to_leader / (np.linalg.norm(vec_to_leader) + 1e-6)

        else:
            centroid = np.mean(positions, axis=0)
            vec_to_group = centroid - pos
            cohesion = vec_to_group / (np.linalg.norm(vec_to_group) + 1e-6)

        # --- Separation ---
        for npos in positions:
            diff = pos - npos
            d = np.linalg.norm(diff)
            if 1e-6 < d < DESIRED_DISTANCE:
                separation += diff / d ** 2

        # --- Alignment ---
        mean_vel = np.mean(velocities, axis=0)
        alignment = mean_vel / (np.linalg.norm(mean_vel) + 1e-6)

    control = (
            COHESION_WEIGHT * cohesion +
            SEPARATION_WEIGHT * separation +
            ALIGNMENT_WEIGHT * alignment
    )

    return control / (np.linalg.norm(control) + 1e-6)


def _cluster_points(points: ndarray, distance_thresh: float = 0.25) -> list[ndarray]:
    """Group nearby LiDAR points into clusters."""
    if len(points) == 0:
        return []

    dist_matrix: ndarray = squareform(pdist(points))
    visited: ndarray = np.zeros(len(points), dtype=bool)
    clusters: list[ndarray] = []

    for i in range(len(points)):
        if visited[i]:
            continue

        cluster: list[int] = [i]
        visited[i] = True
        stack: list[int] = [i]
        while stack:
            idx: int = stack.pop()
            neighbors: ndarray = np.where(dist_matrix[idx] < distance_thresh)[0]
            for n in neighbors:
                if not visited[n]:
                    visited[n] = True
                    cluster.append(n)
                    stack.append(n)

        clusters.append(points[cluster])

    return clusters


def _estimate_leader_from_lidar(lidar_points: ndarray) -> Optional[ndarray]:
    """Detect rectangular object (leader) using LiDAR clusters."""
    if len(lidar_points) == 0:
        return None

    clusters: list[ndarray] = _cluster_points(lidar_points[:, :2])

    leader_cluster: Optional[ndarray] = None
    max_box_area: float = 0

    for c in clusters:
        if len(c) < 3:
            continue

        xs: ndarray = c[:, 0]
        ys: ndarray = c[:, 1]
        width: float = xs.max() - xs.min()
        height: float = ys.max() - ys.min()
        box_area: float = width * height

        if width > 0.4 or height > 0.4:
            if box_area > max_box_area:
                leader_cluster = c
                max_box_area = box_area

    if leader_cluster is None:
        return None

    centroid: ndarray = np.mean(leader_cluster, axis=0)
    return centroid


def _compute_lidar_points(pos, th, lidar) -> ndarray:
    """Compute lidar points."""
    # Build neighbor list
    scan_data: dict = lidar.get_scan()
    ranges: ndarray = np.array(scan_data["ranges"])
    angles: ndarray = np.linspace(lidar.angle_min, lidar.angle_max, len(ranges))
    velocities: ndarray = np.array(scan_data["velocity"])

    lidar_points: list[list[float]] = []
    for r, ang, vx, vy in zip(ranges, angles, velocities[0], velocities[1]):
        if LIDAR_OFFSET < r < lidar.range_max - LIDAR_OFFSET:
            # Compute position in world frame
            x = pos[0] + r * np.cos(ang + th)
            y = pos[1] + r * np.sin(ang + th)

            # Rotate local velocity (vx, vy) into world frame
            wx = vx * np.cos(th) - vy * np.sin(th)
            wy = vx * np.sin(th) + vy * np.cos(th)

            lidar_points.append([x, y, wx, wy])

    return np.array(lidar_points)


def _update_robot(ego_object):
    x, y, th = _to_floats(ego_object.state)
    _ROBOT_POS[ego_object.name] = [x, y, th]


def distance(robot_name):
    x1, y1, _ = _ROBOT_POS[LEADER_NAME]
    x2, y2, _ = _ROBOT_POS[robot_name]
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def print_final_efficiency():
    num_followers = len(_ROBOT_POS) - 1
    median = sorted(FOLLOWER_DIST)[len(FOLLOWER_DIST) // 2] \
        if len(FOLLOWER_DIST) % 2 \
        else (sorted(FOLLOWER_DIST)[len(FOLLOWER_DIST) // 2 - 1] + sorted(FOLLOWER_DIST)[len(FOLLOWER_DIST) // 2]) / 2

    print("\n======================================")
    print(" FINAL EFFICIENCY RESULTS")
    print(f" Followers:                       {num_followers}")
    print(f" Steps used (aligned):            {len(FOLLOWER_DIST) / num_followers}")
    print(f" Total leader–follower pairs:     {len(FOLLOWER_DIST)}")
    print(f" Mean distance from leader:       {sum(FOLLOWER_DIST) / len(FOLLOWER_DIST):.3f}")
    print(f" Median distance from leader:     {median:.3f}")
    print(f" Min-Max distance from leader:    {min(FOLLOWER_DIST):.3f} - {max(FOLLOWER_DIST):.3f}")
    print("======================================\n")

    FOLLOWER_DIST.clear()


# --- Leader behavior ---
@register_behavior('diff', 'custom_behaviour')
def move(ego_object, objects=None, **kw) -> ndarray:
    _update_robot(ego_object)

    if ego_object.name != LEADER_NAME:
        FOLLOWER_DIST.append(distance(ego_object.name))

    x, y, th = _to_floats(ego_object.state)
    pos: ndarray = np.array([x, y])
    lidar_points: ndarray = _compute_lidar_points(pos, th, ego_object.lidar)

    # Estimate leader
    leader_est: Optional[ndarray] = (
        np.array([GOAL_POS['x'], GOAL_POS['y']])
        if ego_object.name == 'robot_0'
        else _estimate_leader_from_lidar(lidar_points)
    )

    # Compute control (leader if seen, otherwise flock)
    control_vec: ndarray = _boid_control(pos, lidar_points, leader_pos=leader_est)

    desired_yaw: float = float(np.arctan2(control_vec[1], control_vec[0]))

    linear_velocity = (
        MAX_SPEED
        if leader_est is None
        else MAX_SPEED * max(0, min(1, np.linalg.norm(leader_est - pos) - STOPPING_DISTANCE))
    )
    angular_velocity: float = ANGULAR_GAIN * np.arctan2(
        np.sin(desired_yaw - th), np.cos(desired_yaw - th)
    )

    return np.array(
        [
            [linear_velocity if ego_object.name != LEADER_NAME else linear_velocity * 0.75],
            [min(max(angular_velocity, -1), 1)]
        ], dtype=float
    )
