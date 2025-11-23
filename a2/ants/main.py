import irsim

from custom_behaviour import GOAL_POS, _PHERO

# TRAIN
env = irsim.make('mas_leader_follower.yaml')
env.load_behavior('custom_behaviour')

UPDATE_EVERY = 15  # update pheromone display this often (env steps)

for t in range(180_0):
    mouse_pos = env.mouse.mouse_pos
    if mouse_pos is not None:
        GOAL_POS['x'], GOAL_POS['y'] = mouse_pos

    env.step()

    # update pheromone visualization less often (every UPDATE_EVERY steps) 
    if (t % UPDATE_EVERY) == 0: _PHERO.render(cmap='hot')

    env.render(0.025)

env.end(1)
