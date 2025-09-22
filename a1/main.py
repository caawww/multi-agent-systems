import irsim

# TRAIN
env = irsim.make('mas_circle_train.yaml')
env.load_behavior('custom_train_and_test')

for _ in range(20 * 200):
    env.step()
    # env.render(0.001)

env.end(1)

# TEST
env = irsim.make('mas_circle_test.yaml')
env.load_behavior('custom_train_and_test')

for _ in range(100 * 10):
    env.step()
    env.render(0.05)

env.end()
