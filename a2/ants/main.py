import irsim
from custom_behaviour import print_final_efficiency, reset_efficiency

# TRAIN
validruns = 0
res_cum = 0
NUM_RUNS = 10
while validruns < NUM_RUNS:
    reset_efficiency()
    env = irsim.make('mas_leader_follower.yaml')
    env.load_behavior('custom_behaviour')

    for _ in range(300):
        env.step()
        env.render(0.001)

    env.end(1)

    # 👇 This will now print the final average follower distance
    res = print_final_efficiency()
    if res:
        res_cum = res_cum + res
        validruns = validruns + 1

print(f"Average following distance: {(res_cum/NUM_RUNS):.3f}")