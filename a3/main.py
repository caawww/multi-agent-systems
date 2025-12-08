import irsim

from config import regenerate_yaml, MAX_STEPS, EPOCHS

regenerate_yaml('train')
env = irsim.make('grid.yaml')
env.load_behavior('custom_behaviour')

for _ in range(EPOCHS * MAX_STEPS):
    env.step()

env.end()

#regenerate_yaml('test')
env = irsim.make('grid.yaml')
env.load_behavior('custom_behaviour')

for _ in range(20_0):
    env.step()
    env.render(0.2)

env.end()
