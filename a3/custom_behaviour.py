import numpy as np
from irsim.lib import register_behavior

from config import GRID_X, GRID_Y

ACTIONS = [
    (1, 0),  # right
    (-1, 0),  # left
    (0, 1),  # up
    (0, -1),  # down
    (0, 0),  # stay
    (0, 0)  # collect
]

EPS_INIT = 0.2
EPS_DECAY = 0.985
ALPHA = 0.1
GAMMA = 0.99
MAX_STEPS = 100

STEP_PENALTY = -0.5
APPLE_REWARD = 500
FALSE_COLLECT = -5

ROBOTS = {}
ROBOTS_REF = set()
REMAINING_APPLES = {}
REMAINING_APPLES_REF = set()


# -------------------------
# Apple class
# -------------------------
class Apple:
    def __init__(self, ego_object):
        self.ego_object = ego_object
        self.init_pose = [ego_object.state[0][0], ego_object.state[1][0]]
        self.pose = self.init_pose.copy()

    def reset(self):
        self.pose = self.init_pose.copy()
        self.ego_object.set_state([self.pose[0], self.pose[1], 0])


# -------------------------
# Robot class
# -------------------------
class Robot:
    def __init__(self, ego_object, x, y, th):
        self.ego_object = ego_object
        self.pose = [x, y, th]
        self.init_pose = [x, y, th]
        self.eps = EPS_INIT
        self.steps = 0
        self.episodes = 0
        self.overall_reward = 0
        self.Q_table = {}

    def get_state(self):
        x, y = self.pose[0], self.pose[1]
        dx, dy = _closest_apple_vector(x, y)
        return x, y, dx, dy

    def choose_action(self, valid_actions):
        state = self.get_state()
        if state not in self.Q_table:
            self.Q_table[state] = np.zeros(len(ACTIONS))

        if np.random.rand() < self.eps:
            return np.random.choice(valid_actions)

        else:
            q_values = self.Q_table[state]
            best_actions = [a for a in valid_actions if q_values[a] == np.max(q_values[valid_actions])]
            return np.random.choice(best_actions)

    def update_q(self, state, action, reward, next_state):
        if next_state not in self.Q_table:
            self.Q_table[next_state] = np.zeros(len(ACTIONS))

        self.Q_table[state][action] += ALPHA * (
                reward + GAMMA * np.max(self.Q_table[next_state]) - self.Q_table[state][action]
        )

    def move(self, a, reward):
        dx, dy = ACTIONS[a]
        nx, ny = self.pose[0] + dx, self.pose[1] + dy

        self.pose[0] = nx
        self.pose[1] = ny
        self.steps += 1
        self.overall_reward += reward

        return self.pose

    def reset(self):
        self.print_stats()

        self.pose = self.init_pose.copy()
        self.ego_object.set_state([self.pose[0], self.pose[1], 0])
        self.steps = 0
        self.episodes += 1
        self.overall_reward = 0
        self.eps *= EPS_DECAY

    def print_stats(self):
        print(
            f'[train] {self.ego_object.name:9} steps:{self.steps:3}, reward:{self.overall_reward:7}, episodes:{self.episodes:3}, eps:{round(self.eps, 3):6}')


def _to_floats(pose):
    return [float(p) for p in pose]


def _valid(nx, ny):
    return 0 <= nx < GRID_X and 0 <= ny < GRID_Y and \
        np.all([[nx, ny] != apple.pose for apple in REMAINING_APPLES.values()])


def _valid_actions(x, y):
    valid = [i for i, (dx, dy) in enumerate(ACTIONS) if _valid(x + dx, y + dy)]

    if _check_adjacent_apple(x, y) is None:
        valid.remove(5)

    return valid


def _closest_apple_vector(x, y):
    if not REMAINING_APPLES:
        return 0, 0

    # find apple with minimum manhattan distance
    best_dx, best_dy = 0, 0
    best_dist = float("inf")

    for apple in REMAINING_APPLES.values():
        ax, ay = apple.pose
        dx, dy = ax - x, ay - y
        dist = abs(dx) + abs(dy)
        if dist < best_dist:
            best_dist = dist
            best_dx, best_dy = dx, dy

    return best_dx, best_dy


def _get_robot_for(ego_object):
    name = ego_object.name
    if name not in ROBOTS:
        pose = _to_floats(ego_object.state)
        ROBOTS[name] = Robot(ego_object, pose[0], pose[1], pose[2])

    return ROBOTS[name]


def _check_adjacent_apple(x, y):
    for a in ACTIONS[:4]:
        nx, ny = x + a[0], y + a[1]
        for key, apple in REMAINING_APPLES.items():
            if apple.pose == [nx, ny]:
                return key

    return None


def _collect_apple(apple_name):
    if apple_name in REMAINING_APPLES:
        REMAINING_APPLES[apple_name].ego_object.set_state([-10, -10, 0])
        del REMAINING_APPLES[apple_name]
        return True

    return False


@register_behavior('diff', 'train')
def train(ego_object, objects=None, **kw):
    robot = _get_robot_for(ego_object)
    ROBOTS_REF.add(robot)
    state = robot.get_state()

    valid_actions = _valid_actions(*robot.pose[:2])
    action = robot.choose_action(valid_actions)

    reward = STEP_PENALTY

    if action == 5:  # collect apple
        apple_name = _check_adjacent_apple(*robot.pose[:2])
        if apple_name:
            _collect_apple(apple_name)
            reward += APPLE_REWARD

        else:
            reward += FALSE_COLLECT

    robot.move(action, reward)
    next_state = robot.get_state()
    robot.update_q(state, action, reward, next_state)

    ego_object.set_state(robot.pose)

    if robot.steps >= MAX_STEPS:  # or len(REMAINING_APPLES) == 0
        # for r in ROBOTS_REF:
        #     r.reset()
        robot.reset()

        for a in REMAINING_APPLES_REF:
            a.reset()

    return np.array([[0.0], [0.0]], dtype=float)


@register_behavior('diff', 'test')
def test(ego_object, objects=None, **kw):
    robot = _get_robot_for(ego_object)
    state = robot.get_state()

    valid_actions = _valid_actions(*robot.pose[:2])

    if state not in robot.Q_table:
        action = np.random.choice(valid_actions)

    else:
        q_values = robot.Q_table[state]
        best_actions = [a for a in valid_actions if q_values[a] == np.max(q_values[valid_actions])]
        action = np.random.choice(best_actions)

    reward = STEP_PENALTY

    if action == 5:
        apple_name = _check_adjacent_apple(*robot.pose[:2])
        if apple_name:
            _collect_apple(apple_name)
            reward += APPLE_REWARD

        else:
            reward += FALSE_COLLECT

    else:
        robot.move(action, reward)

    ego_object.set_state(robot.pose)
    return np.array([[0.0], [0.0]], dtype=float)


@register_behavior('diff', 'apple')
def apple(ego_object, objects=None, **kw):
    if ego_object.state[0][0] < -5:
        return np.array([[0.0], [0.0]], dtype=float)

    if ego_object.name not in REMAINING_APPLES:
        apple_obj = Apple(ego_object)
        REMAINING_APPLES[ego_object.name] = apple_obj
        REMAINING_APPLES_REF.add(apple_obj)

    return np.array([[0.0], [0.0]], dtype=float)
