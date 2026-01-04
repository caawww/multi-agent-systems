import numpy as np

from config import ACTIONS, EPS_INIT, REMAINING_APPLES, ALPHA, GAMMA, EPS_DECAY


class Apple:
    def __init__(self, ego_object, level=1):
        self.ego_object = ego_object
        self.init_pose = [ego_object.state[0][0], ego_object.state[1][0]]
        self.pose = self.init_pose.copy()
        self.level = level

    def reset(self):
        self.pose = self.init_pose.copy()
        self.ego_object.set_state([self.pose[0], self.pose[1], 0])


class Robot:
    def __init__(self, ego_object, x, y, th, level=1):
        self.ego_object = ego_object
        self.pose = [x, y, th]
        self.init_pose = [x, y, th]
        self.eps = EPS_INIT
        self.steps = 0
        self.episodes = 0
        self.overall_reward = 0
        self.Q_table = {}
        self.level = level

    def get_state(self):
        x, y = self.pose[0], self.pose[1]
        dx, dy, apple_level = self._closest_apple_info(x, y)
        return dx, dy, apple_level

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

    def _closest_apple_info(self, x, y):
        if not REMAINING_APPLES:
            return 0, 0, 0

        best_dx, best_dy = 0, 0
        best_level = 0
        best_dist = float('inf')

        for apple in REMAINING_APPLES.values():
            ax, ay = apple.pose
            dx, dy = ax - x, ay - y
            dist = abs(dx) + abs(dy)
            if dist < best_dist:
                best_dist = dist
                best_dx, best_dy = dx, dy
                best_level = apple.level

        return best_dx, best_dy, best_level

    def print_stats(self):
        print(
            f'[train] {self.ego_object.name:9} steps:{self.steps:4}, reward:{self.overall_reward:7}, episodes:{self.episodes:3}, eps:{round(self.eps, 3):6}')
