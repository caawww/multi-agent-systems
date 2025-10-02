import irsim
from custom_train_and_test import TRAINING_STEPS, TESTING_STEPS

# TRAIN
env = irsim.make('mas_circle_train.yaml')
env.load_behavior('custom_train_and_test')

for _ in range(5 * TRAINING_STEPS):
    env.step()
    # env.render(0.001)

env.end(1)

# TEST
env = irsim.make('mas_circle_test.yaml')
env.load_behavior('custom_train_and_test')

for _ in range(1 + TESTING_STEPS):
    env.step()
    env.render(0.001)

env.end()
