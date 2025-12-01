import numpy as np
from irsim.lib import register_behavior

from config import GRID_X, GRID_Y

ACTIONS = [
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (0, 0),
    (0, 0)
]

ROBOTS = {}
REMAINING_APPLES = {}
EPS_INIT = 0.2
EPS_DECAY = 0.985


def _to_floats(pose):
    return [float(p) for p in pose]


def _valid(nx, ny):
    return (
            0 <= nx < GRID_X
            and 0 <= ny < GRID_Y
            and np.all([[nx, ny] != key[1] for key in REMAINING_APPLES.values()])
    )


def _contains_apple(x, y):
    for key in REMAINING_APPLES:
        if REMAINING_APPLES[key][1] == [x, y]:
            return key

    return None


def _check_for_apple(x, y):
    for a in ACTIONS[:4]:
        apple = _contains_apple(x + a[0], y + a[1])
        if apple is not None:
            return apple

    return None


def _collect_apple(apple_name):
    if apple_name in REMAINING_APPLES:
        REMAINING_APPLES[apple_name][0].set_state([-10, -10, 1])
        return True

    return False


def _valid_actions(x, y):
    return [i for i, (dx, dy) in enumerate(ACTIONS) if _valid(x + dx, y + dy)]


def _get_robot_for(ego_object):
    name = ego_object.name

    if name not in ROBOTS:
        pose = _to_floats(ego_object.state)
        ROBOTS[name] = Robot(pose[0], pose[1], pose[2])

    return ROBOTS[name]


class Robot:
    def __init__(self, x, y, th, eps=EPS_INIT):
        self.pose = [x, y, th]
        self.init_pose = [x, y, th]
        self.eps = eps
        self.steps = 0
        self.episodes = 0
        self.overall_reward = 0

    def move(self, a):
        dx, dy = ACTIONS[a]
        nx, ny = self.pose[0] + dx, self.pose[1] + dy

        if not _valid(nx, ny):
            raise RobotError(f"Invalid robot position: ({nx}, {ny})")

        self.pose[0] = nx
        self.pose[1] = ny
        return self.pose


class RobotError(Exception):
    pass


@register_behavior('diff', 'train')
def train(ego_object, objects=None, **kw):
    robot = _get_robot_for(ego_object)

    acts = _valid_actions(*robot.pose[:2])
    a = np.random.choice(acts)
    robot.move(a)

    ego_object.set_state(robot.pose)

    if a == 5:  # collect action
        is_apple_collected = _collect_apple(_check_for_apple(*robot.pose[:2]))

    return np.array([[0.0], [0.0]], dtype=float)


@register_behavior('diff', 'test')
def test(ego_object, objects=None, **kw):
    return np.array([[0.0], [0.0]], dtype=float)


@register_behavior('diff', 'apple')
def apple(ego_object, objects=None, **kw):
    if ego_object.state[0][0] < -5:
        return np.array([[0.0], [0.0]], dtype=float)

    if ego_object.name not in REMAINING_APPLES:
        REMAINING_APPLES[ego_object.name] = [ego_object, _to_floats(ego_object.state)[:2]]

    return np.array([[0.0], [0.0]], dtype=float)
