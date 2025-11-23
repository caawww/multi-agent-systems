import irsim

from custom_behaviour import GOAL_POS, INTERACTIVE, print_final_efficiency

if INTERACTIVE:
    env = irsim.make('mas_leader_follower.yaml')
    env.load_behavior('custom_behaviour')

    for _ in range(180_0):
        mouse_pos = env.mouse.mouse_pos
        if mouse_pos is not None:
            GOAL_POS['x'], GOAL_POS['y'] = mouse_pos

        env.step()
        env.render(0.025)

    env.end(1)

else:
    for i in range(10):
        env = irsim.make('mas_leader_follower.yaml')
        env.load_behavior('custom_behaviour')

        for _ in range(30_0):
            env.step()
            # env.render(0.025)

        env.end(1)
        print_final_efficiency()
