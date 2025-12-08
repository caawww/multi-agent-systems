import random

from create_grid import generate_yaml

IS_RANDOMIZED = True

ROBOTS = dict()
ROBOTS_REF = set()
REMAINING_APPLES = dict()
REMAINING_APPLES_REF = set()

EPOCHS = 50
MAX_STEPS = 100

TIME_PENALTY = -1
STEP_PENALTY = -1
APPLE_REWARD = 1000

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

ROBOTS_POS = [
    {'state': (0, 0, 0), 'level': 10, 'radius': 0.2},
    {'state': (4, 0, 0), 'level': 15, 'radius': 0.2},
    {'state': (9, 0, 0), 'level': 15, 'radius': 0.2},
    # {'state': (2, 2, 0), 'level': 20, 'radius': 0.2},
    # {'state': (3, 3, 0), 'level': 20, 'radius': 0.2},
    # {'state': (4, 4, 0), 'level': 20, 'radius': 0.2},
]

APPLES_POS = list(set((random.randrange(10), random.randrange(9) + 1) for _ in range(10)))

ROBOT_COUNT = random.randint(2, 5)
APPLES_COUNT = random.randint(3, 10)
GRID_X = random.randint(5, 15) if IS_RANDOMIZED else 10
GRID_Y = random.randint(5, 15) if IS_RANDOMIZED else 10
MAX_LEVEL = random.randint(2, 5)

RANDOM_ROBOTS_POS = list(
    {
        'state': (random.randrange(0, GRID_X), random.randrange(0, GRID_Y), 0),
        'level': MAX_LEVEL,
        'radius': 0.2
    }
    for _ in range(ROBOT_COUNT)
)

RANDOM_APPLES_POS = list(
    set(
        (random.randrange(0, GRID_X), random.randrange(0, GRID_Y))
        for _ in range(APPLES_COUNT)
    )
)


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
