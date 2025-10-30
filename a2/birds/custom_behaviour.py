import numpy as np
import random

from fontTools.subset import prune_hints
from irsim.lib import register_behavior

# --- Parameters ---
MAX_SPEED = 0.15
ANGULAR_GAIN = 2.0
COHESION_WEIGHT = 0.6
SEPARATION_WEIGHT = 1.2
ALIGNMENT_WEIGHT = 0.4
DESIRED_DISTANCE = 0.8

LEADER_POS = {
    'x': 0.0,
    'y': 0.0,
    'th': 0.0
}


def _to_floats(pos):
    """Convert pos list to plain floats."""
    return [float(p) for p in pos]


# --- Helper functions ---
def _boid_control(pos, heading, neighbors, leader_pos):
    # --- Cohesion: move toward leader ---
    vec_to_leader = leader_pos - pos
    cohesion = vec_to_leader / (np.linalg.norm(vec_to_leader) + 1e-6)

    # --- Separation: avoid nearby objects ---
    separation = np.zeros(2)
    for npos, _ in neighbors:
        diff = pos - npos
        d = np.linalg.norm(diff)
        if 1e-6 < d < DESIRED_DISTANCE:
            separation += diff / d ** 2

    # --- Alignment: align heading with neighbors ---
    alignment = np.zeros(2)
    if len(neighbors) > 0:
        neighbor_dirs = [nvel for _, nvel in neighbors]
        mean_dir = np.mean(neighbor_dirs, axis=0)
        alignment = mean_dir / (np.linalg.norm(mean_dir) + 1e-6)

    # Combine
    control = (COHESION_WEIGHT * cohesion +
               SEPARATION_WEIGHT * separation +
               ALIGNMENT_WEIGHT * alignment)

    return control / (np.linalg.norm(control) + 1e-6)


# --- Leader behavior ---
@register_behavior('diff', 'custom_behaviour')
def leader(ego_object, objects=None, **kw):
    if ego_object.name == 'robot_0':
        x, y, _ = _to_floats(ego_object.state)
        LEADER_POS['x'] = x
        LEADER_POS['y'] = y

        return np.array(
            [
                [MAX_SPEED * random.random() if y < 5 else 0],  # linear velocity
                [0.0]  # angular velocity
            ],
            dtype=float
        )

    x, y, th = _to_floats(ego_object.state)
    pos = np.array([x, y])
    heading = np.array([np.cos(th), np.sin(th)])
    neighbors = []

    # --- LiDAR perception ---
    lidar = ego_object.lidar
    scan_data = lidar.get_scan()  # returns dict: {angles, ranges, velocities}
    ranges = np.array(scan_data["ranges"])
    angles = np.linspace(lidar.angle_min, lidar.angle_max, len(ranges))

    for r, ang in zip(ranges, angles):
        if 0.05 < r < lidar.range_max:
            rel = np.array([r * np.cos(ang + th), r * np.sin(ang + th)])
            neighbor_pos = pos + rel
            neighbor_vel = np.zeros(2)
            neighbors.append((neighbor_pos, neighbor_vel))

    # Find leader
    leader_pos = np.array([LEADER_POS['x'], LEADER_POS['y']])

    # Compute control
    control_vec = _boid_control(pos, heading, neighbors, leader_pos)
    desired_yaw = np.arctan2(control_vec[1], control_vec[0])

    angular_velocity = ANGULAR_GAIN * np.arctan2(np.sin(desired_yaw - th), np.cos(desired_yaw - th))

    # print(np.linalg.norm(leader_pos - pos))

    # Control velocities
    return np.array(
        [
            [MAX_SPEED],  # linear velocity
            [min(max(angular_velocity, -1), 1)]  # angular velocity
        ],
        dtype=float
    )
