import irsim

from config import regenerate_yaml

regenerate_yaml()

env = irsim.make('grid.yaml')
env.load_behavior('custom_behaviour')

for _ in range(150 * 10_0):
    env.step()
    # env.render(0.01)

env.end()

regenerate_yaml('test')

env = irsim.make('grid.yaml')
env.load_behavior('custom_behaviour')

for _ in range(120_0):
    env.step()
    env.render(0.5)

env.end()
