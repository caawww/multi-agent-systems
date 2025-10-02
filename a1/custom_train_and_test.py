import numpy as np
from irsim.lib import register_behavior

# Circle definition
CENTER = (5.0, 5.0)
RADIUS = 4

# Params
ALPHA = 0.5
GAMMA = 0.95
EPS_INIT = 0.2
EPS_MIN = 0.01
EPS_DECAY = 0.995

# Actions: forward, turn left, turn right
ACTIONS = [
    (0.05, 0.0),  # forward
    (0.05, 0.3),  # turn left
    (0.05, -0.3),  # turn right
]

TRAINING_STEPS = 200
TESTING_STEPS = 500

# Global robot data
_robot_data = {}  # ego_name -> dict
_Q = {}  # state -> q-values


def _discretize(pose):
    """Discretization using distance and heading errors."""
    x, y, th = pose
    vec = np.array([x, y]) - CENTER
    dist = np.linalg.norm(vec)
    dist_err = round(dist - RADIUS, 1)  # Distance error from circle

    tangent = np.arctan2(vec[1], vec[0]) + np.pi / 2
    heading_err = ((th - tangent + np.pi) % (2 * np.pi)) - np.pi
    heading_err = round(heading_err, 1)  # Heading error from tangent

    return (dist_err, heading_err)


def _to_floats(pose):
    """Convert pose list to plain floats."""
    return [float(p) for p in pose]


def _ensure(s):
    if s not in _Q:
        _Q[s] = np.zeros(len(ACTIONS), dtype=float)


def _reward(pose, dist=True, head=True):
    """Reward is based on distance to the circle outline."""
    x, y, th = pose
    vec = np.array([x, y]) - CENTER

    dist_error = 0
    if dist:
        # Distance error to circle outline
        dist_to_center = np.linalg.norm(vec)
        dist_error = abs(dist_to_center - RADIUS)

    heading_err = 0
    if head:
        # Heading alignment with tangent
        tangent = np.arctan2(vec[1], vec[0]) + np.pi / 2
        heading_err = abs(((th - tangent + np.pi) % (2 * np.pi)) - np.pi)

    # Final reward (negative because we want to minimize error)
    return - (dist_error + 0.5 * heading_err)


def _step_pose(pose, action):
    """Move robot according to action."""
    x, y, th = pose
    v, w = action
    th_new = th + w
    x_new = x + v * np.cos(th_new)
    y_new = y + v * np.sin(th_new)
    return [x_new, y_new, th_new]


def _init_robot(ego_object):
    x, y, th = _to_floats(ego_object.state)
    _robot_data[ego_object.name] = {
        'pose': [x, y, th],
        'init_pose': [x, y, th],
        'eps': EPS_INIT,
        'steps': 0,
        'episodes': 0,
        'overall_reward': 0,
    }


# TRAIN ------------------------------------------------------------------------------------------------
@register_behavior('diff', 'train')
def train(ego_object, objects=None, **kw):
    if ego_object.name not in _robot_data:
        _init_robot(ego_object)

    data = _robot_data[ego_object.name]
    pose = data['pose']
    state = _discretize(pose)

    # Epsilon-greedy action
    if (np.random.rand() < data['eps']) or (state not in _Q):
        a = np.random.choice(len(ACTIONS))  # explore
    else:
        _ensure(state)
        a = int(np.argmax(_Q[state]))  # exploit

    # Take action
    new_pose = _step_pose(pose, ACTIONS[a])
    new_state = _discretize(new_pose)

    # Reward
    reward = _reward(new_pose)
    data['overall_reward'] += reward

    # Q-learning update
    _ensure(state)
    _ensure(new_state)
    best_next = np.max(_Q[new_state])
    target = reward + GAMMA * best_next
    _Q[state][a] += ALPHA * (target - _Q[state][a])

    # Update robot state
    data['pose'] = new_pose
    data['steps'] += 1
    ego_object.set_state(new_pose)

    # Episode management: reset every `n` steps
    if data['steps'] >= TRAINING_STEPS:
        data['steps'] = 0
        data['episodes'] += 1
        data['eps'] = max(EPS_MIN, data['eps'] * EPS_DECAY)
        ego_object.set_state(data['init_pose'])
        data['pose'] = list(data['init_pose'])

        # print(
        #     f"[train] {ego_object.name} finished episode={data['episodes']:3}, "
        #     f"eps={data['eps']:6.3f}, "
        #     f"reward={data['overall_reward']:5.0f}"
        # )

        data['overall_reward'] = 0

    return np.array([[0.0], [0.0]], dtype=float)


# TEST ------------------------------------------------------------------------------------------------
@register_behavior('diff', 'test')
def test(ego_object, objects=None, **kw):
    if ego_object.name not in _robot_data:
        _init_robot(ego_object)

    data = _robot_data[ego_object.name]
    if data['steps'] >= TESTING_STEPS:
        if data['steps'] == TESTING_STEPS:
            print(
                f"[test] {ego_object.name} finished, "
                f"train episodes={data['episodes']:3}, "
                f"reward={data['overall_reward']:8.2f}"
            )

        return np.array([[0.0], [0.0]], dtype=float)

    pose = data['pose']
    state = _discretize(pose)

    _ensure(state)
    a = int(np.argmax(_Q[state]))
    new_pose = _step_pose(pose, ACTIONS[a])

    data['pose'] = new_pose
    ego_object.set_state(new_pose)

    data['overall_reward'] += _reward(new_pose, head=False)
    data['steps'] += 1

    return np.array([[0.0], [0.0]], dtype=float)
