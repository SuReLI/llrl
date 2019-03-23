from llrl.envs.gridworld import GridWorld
from llrl.algorithms.dynamic_programming import dynamic_programming

# Parameters
env = GridWorld(map_name='frozen_lake', slipperiness=.5)
gamma = 0.9
threshold = 1e-10
iter_max = 1000

value_function = dynamic_programming(env, gamma=gamma, threshold=threshold, iter_max=iter_max)

env.render()
env.display_to_m(value_function)
