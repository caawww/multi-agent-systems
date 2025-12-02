import numpy as np
from irsim.lib import register_behavior

from config import ROBOTS, ROBOTS_REF, REMAINING_APPLES, REMAINING_APPLES_REF, MAX_STEPS, TIME_PENALTY, STEP_PENALTY, \
    APPLE_REWARD, ACTIONS, GRID_X, GRID_Y
from entities import Apple, Robot


def _to_floats(pose):
    return [float(p) for p in pose]


def _occupied_by_robot(x, y):
    for r in ROBOTS.values():
        if r.pose[0] == x and r.pose[1] == y:
            return True

    return False


def _valid(nx, ny):
    if not (0 <= nx < GRID_X and 0 <= ny < GRID_Y):
        return False

    for apple in REMAINING_APPLES.values():
        if apple.pose == [nx, ny]:
            return False

    # cannot walk on other robots
    if _occupied_by_robot(nx, ny):
        return False

    return True


def _valid_actions(x, y):
    valid = [i for i, (dx, dy) in enumerate(ACTIONS[:4]) if _valid(x + dx, y + dy)]
    valid.append(4)

    if _check_adjacent_apple(x, y):
        valid.append(5)

    return valid


def _get_robot_for(ego_object):
    name = ego_object.name
    if name not in ROBOTS:
        pose = _to_floats(ego_object.state)
        ROBOTS[name] = Robot(ego_object, pose[0], pose[1], pose[2])

    return ROBOTS[name]


def _adjacent_robots(x, y):
    robots_adj = []
    for dx, dy in ACTIONS[:4]:
        nx, ny = x + dx, y + dy
        for robot in ROBOTS.values():
            if robot.pose[:2] == [nx, ny]:
                robots_adj.append(robot)
    return robots_adj


def _check_adjacent_apple(x, y):
    for dx, dy in ACTIONS[:4]:
        nx, ny = x + dx, y + dy

        for key, apple in REMAINING_APPLES.items():
            if apple.pose == [nx, ny]:
                # sum levels of adjacent robots to this apple
                robots_adj = _adjacent_robots(*apple.pose)
                total_level = sum(r.level for r in robots_adj)
                if total_level >= apple.level:
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

    robot.move(action, reward)
    next_state = robot.get_state()
    robot.update_q(state, action, reward, next_state)

    ego_object.set_state(robot.pose)

    if robot.steps >= MAX_STEPS:  # or len(REMAINING_APPLES) == 0
        # for r in ROBOTS_REF:
        #     r.reset()
        robot.reset()

        for apple_obj in REMAINING_APPLES_REF:
            apple_obj.reset()
            REMAINING_APPLES[apple_obj.ego_object.name] = apple_obj

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
