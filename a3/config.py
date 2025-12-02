import random

from create_grid import generate_yaml

GRID_X = 10
GRID_Y = 10
MAX_STEPS = 200
EPOCHS = 50

ROBOTS = [
    {'state': (0, 0, 0), 'level': 10, 'radius': 0.2},
    {'state': (4, 0, 0), 'level': 15, 'radius': 0.2},
    {'state': (9, 0, 0), 'level': 15, 'radius': 0.2},
    # {'state': (2, 2, 0), 'level': 20, 'radius': 0.2},
    # {'state': (3, 3, 0), 'level': 20, 'radius': 0.2},
    # {'state': (4, 4, 0), 'level': 20, 'radius': 0.2},
]

APPLES = list(set((random.randrange(10), random.randrange(9) + 1) for _ in range(10)))


def regenerate_yaml(behaviour='train'):
    generate_yaml(
        world_size=(GRID_X, GRID_Y),
        robots=ROBOTS,
        apples=APPLES,
        behaviour=behaviour,
        output_file='grid.yaml'
    )
