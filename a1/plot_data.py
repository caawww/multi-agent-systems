import re
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend
import matplotlib.pyplot as plt


# --- Parse log file ---
logfile = "log.txt"

# Regex to match lines like:
# [test] robot_0 finished, train episodes=  1, reward=  -35.46
pattern = re.compile(
    r"\[test\] (robot_\d+) finished, train episodes=\s*(\d+), reward=\s*([-\d.]+)"
)

# Store results as: robot -> list of (episode, reward)
robot_data = defaultdict(list)

with open(logfile, "r") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            robot, episode, reward = match.groups()
            robot_data[robot].append((int(episode), float(reward)))

# --- Plot ---
plt.figure(figsize=(10, 6))

for robot, values in robot_data.items():
    # Sort by episode just in case
    values.sort(key=lambda x: x[0])
    episodes, rewards = zip(*values)
    plt.plot(episodes, rewards, marker="o", label=robot)

plt.title("Robot Rewards Across Episodes")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('fig.pdf')
