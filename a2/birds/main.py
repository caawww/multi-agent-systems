import irsim

# TRAIN
env = irsim.make('mas_leader_follower.yaml')
env.load_behavior('custom_behaviour')

for _ in range(180_0):
    env.step()
    env.render(0.02)

env.end(1)
