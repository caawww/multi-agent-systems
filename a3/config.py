import random

from create_grid import generate_yaml

IS_RANDOMIZED = False

ROBOTS = dict()
ROBOTS_REF = set()
REMAINING_APPLES = dict()
REMAINING_APPLES_REF = set()

GRID_X=10
GRID_Y=10

EPOCHS = 300
MAX_STEPS = 200

TIME_PENALTY = -40
STEP_PENALTY = -5
APPLE_REWARD = 300

ACTIONS = [
    (1, 0),  # right
    (-1, 0),  # left
    (0, 1),  # up
    (0, -1),  # down
    (0, 0),  # stay
    (0, 0)  # collect
]

EPS_INIT = 1.0
EPS_DECAY = 0.995
ALPHA = 0.2
GAMMA = 0.99

ROBOTS_POS = [
    {'state': (5, 0, 0), 'level': 1},
    {'state': (8, 0, 0), 'level': 1},
    {'state': (2, 0, 0), 'level': 1},
    {'state': (4, 0, 0), 'level': 2},
    {'state': (7, 0, 0), 'level': 2},
    {'state': (6, 0, 0), 'level': 2},
   
    # {'state': (2, 2, 0), 'level': 20, 'radius': 0.2},
    # {'state': (3, 3, 0), 'level': 20, 'radius': 0.2},
    # {'state': (4, 4, 0), 'level': 20, 'radius': 0.2},
]

APPLES_POS = [
    {
        'state': (x, y),
        'level': random.choice([1, 2, 3])
    }
    # Constrain apples to spawn in the upper-left corner (e.g., top-left 25% of the grid)
    for (x, y) in set(
        (random.randrange(0, GRID_X // 2), random.randrange(GRID_Y // 2, GRID_Y))
        for _ in range(10)
    )
]

ROBOT_COUNT = random.randint(2, 5)
APPLES_COUNT = random.randint(3, 10)
GRID_X = random.randint(5, 15) if IS_RANDOMIZED else 10
GRID_Y = random.randint(5, 15) if IS_RANDOMIZED else 10
MAX_LEVEL = random.randint(2, 4)

# TODO - check same square placement
RANDOM_ROBOTS_POS = list(
    {
        'state': (random.randrange(0, GRID_X), random.randrange(0, GRID_Y), 0),
        'level': MAX_LEVEL,
        'radius': 0.2
    }
    for _ in range(ROBOT_COUNT)
)

# Ensure the grid size remains fixed
GRID_X = 10
GRID_Y = 10

# Randomize apple positions every training session
RANDOM_APPLES_POS = []
while len(RANDOM_APPLES_POS) < APPLES_COUNT:
    x, y = random.randrange(0, GRID_X), random.randrange(0, GRID_Y)

    for robot in RANDOM_ROBOTS_POS + RANDOM_APPLES_POS:
        rx, ry = robot['state'][:2]
        if rx == x and ry == y:
            break
    else:
        RANDOM_APPLES_POS.append({
            'state': (x, y),
            'level': random.randint(1, MAX_LEVEL),
        })


def create_yaml(behaviour='train'):
    if IS_RANDOMIZED:
        generate_yaml(
            world_size=(GRID_X, GRID_Y),
            robots=RANDOM_ROBOTS_POS,
            apples=RANDOM_APPLES_POS,
            behaviour=behaviour,
            output_file='grid.yaml'
        )

    else:
        generate_yaml(
            world_size=(GRID_X, GRID_Y),
            robots=ROBOTS_POS,
            apples=APPLES_POS,
            behaviour=behaviour,
            output_file='grid.yaml'
        )
