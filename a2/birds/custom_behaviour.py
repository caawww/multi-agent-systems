import random
from typing import Optional

import numpy as np
from irsim.lib import register_behavior
from numpy import ndarray
from scipy.spatial.distance import pdist, squareform

# --- Parameters ---
MAX_SPEED: float = 0.15
ANGULAR_GAIN: float = 2.0
COHESION_WEIGHT: float = 0.6
SEPARATION_WEIGHT: float = 1.2
ALIGNMENT_WEIGHT: float = 0.4
DESIRED_DISTANCE: float = 1.0


def _to_floats(pos: ndarray) -> list[float]:
    """Convert pos list to plain floats."""
    return [float(p) for p in pos]


def _boid_control(pos: ndarray, lidar_points: ndarray, leader_pos: Optional[ndarray] = None) -> ndarray:
    """Boid control with optional leader influence."""

    # --- Cohesion: move toward leader or neighbor centroid ---
    cohesion: ndarray = np.zeros(2)
    if leader_pos is not None:
        vec_to_leader: ndarray = leader_pos - pos
        cohesion = vec_to_leader / (np.linalg.norm(vec_to_leader) + 1e-6)

    elif len(lidar_points) > 0:
        centroid: ndarray = np.mean([npos for npos in lidar_points], axis=0)
        vec_to_group: ndarray = centroid - pos
        cohesion = vec_to_group / (np.linalg.norm(vec_to_group) + 1e-6)

    # --- Separation: avoid nearby objects ---
    separation: ndarray = np.zeros(2)
    for npos in lidar_points:
        diff: ndarray = pos - npos
        d = np.linalg.norm(diff)
        if 1e-6 < d < DESIRED_DISTANCE:
            separation += diff / d ** 2

    # # --- Alignment: align heading with neighbors ---
    # alignment: ndarray = np.zeros(2)
    # if len(lidar_points) > 0:
    #     neighbor_dirs: list = [nvel for _, nvel in lidar_points]
    #     mean_dir: ndarray = np.mean(neighbor_dirs, axis=0)
    #     alignment = mean_dir / (np.linalg.norm(mean_dir) + 1e-6)

    control: ndarray = (
            COHESION_WEIGHT * cohesion
            + SEPARATION_WEIGHT * separation
        # + ALIGNMENT_WEIGHT * alignment
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

    clusters: list[ndarray] = _cluster_points(lidar_points)

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


# --- Leader behavior ---
@register_behavior('diff', 'custom_behaviour')
def move(ego_object, objects=None, **kw) -> ndarray:
    # --- Leader (rectangle) ---
    if ego_object.name == 'robot_0':
        x: float
        y: float
        _: float
        x, y, _ = _to_floats(ego_object.state)
        return np.array(
            [
                [MAX_SPEED * random.random() if y < 5 else 0],
                [0.5 - random.random()],
            ],
            dtype=float
        )

    # --- Follower (circle) ---
    x: float
    y: float
    th: float
    x, y, th = _to_floats(ego_object.state)
    pos: ndarray = np.array([x, y])
    heading: ndarray = np.array([np.cos(th), np.sin(th)])
    lidar = ego_object.lidar

    # Build neighbor list
    lidar_points: list[list[float]] = []
    scan_data: dict = lidar.get_scan()
    ranges: ndarray = np.array(scan_data["ranges"])
    angles: ndarray = np.linspace(lidar.angle_min, lidar.angle_max, len(ranges))

    for r, ang in zip(ranges, angles):
        if 0.05 < r < lidar.range_max - 0.05:
            lidar_points.append([
                x + r * np.cos(ang + th),
                y + r * np.sin(ang + th)
            ])

    lidar_points: ndarray = np.array(lidar_points)

    # Estimate leader
    leader_est: Optional[ndarray] = _estimate_leader_from_lidar(lidar_points)
    if leader_est is not None:
        print(f'{ego_object.name} -> [{leader_est[0]:.2} {leader_est[1]:.2}]')

    # Compute control (leader if seen, otherwise flock)
    control_vec: ndarray = _boid_control(pos, lidar_points, leader_pos=leader_est)

    desired_yaw: float = float(np.arctan2(control_vec[1], control_vec[0]))
    angular_velocity: float = ANGULAR_GAIN * np.arctan2(
        np.sin(desired_yaw - th), np.cos(desired_yaw - th)
    )

    return np.array(
        [
            [MAX_SPEED],
            [min(max(angular_velocity, -1), 1)]
        ],
        dtype=float
    )
