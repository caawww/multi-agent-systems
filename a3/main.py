import irsim

from config import GRID_X, GRID_Y, ROBOTS, APPLES
from create_grid import generate_yaml

generate_yaml(
    world_size=(GRID_X, GRID_Y),
    robots=ROBOTS,
    apples=APPLES,
    output_file="grid.yaml"
)

env = irsim.make('grid.yaml')
env.load_behavior('custom_behaviour')

for _ in range(500):
    env.step()
    env.render(0.01)

env.end()
