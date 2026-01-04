import irsim
import winsound

from config import create_yaml, MAX_STEPS, EPOCHS

create_yaml('train')
env = irsim.make('grid.yaml')
env.load_behavior('custom_behaviour')

for _ in range(EPOCHS*MAX_STEPS):
    #create_yaml('train')
    #for __ in range(MAX_STEPS):

        env.step()

        if(_ > (EPOCHS * MAX_STEPS) - 50 * MAX_STEPS):
            #winsound.Beep(1000, 500)  
            env.render(0.01)

        else: env.render(0.01)

env.end()

create_yaml('test')
env = irsim.make('grid.yaml')
env.load_behavior('custom_behaviour')

for _ in range(20_0):
    env.step()
    env.render(0.2)

env.end()
