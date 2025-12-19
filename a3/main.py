import irsim

from config import create_yaml, MAX_STEPS, EPOCHS

create_yaml('train')
env = irsim.make('grid.yaml')
env.load_behavior('custom_behaviour')

for _ in range(EPOCHS * MAX_STEPS):
    env.step()

env.end()

create_yaml('test')
env = irsim.make('grid.yaml')
env.load_behavior('custom_behaviour')

for _ in range(20_0):
    env.step()
    env.render(0.2)

env.end()
