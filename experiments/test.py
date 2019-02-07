from llrl.envs.gridworld import GridWorld
from llrl.algorithms.dynamic_programming import dynamic_programming
from llrl.envs.handler import Handler
import numpy as np

# Parameters
m1 = GridWorld(map_name='maze1', is_slippery=False)
m2 = GridWorld(map_name='maze2', is_slippery=False)
gamma = 0.9

# Compute
v1 = dynamic_programming(m1, gamma=gamma, threshold=1e-10, iter_max=1000, verbose=False)
v2 = dynamic_programming(m2, gamma=gamma, threshold=1e-10, iter_max=1000, verbose=False)
dv = abs(v1 - v2)

h = Handler()
state_distances = h.bi_simulation_distance(m1, m2)
distance_m1_m2 = h.mdp_distance(m1, m2, state_distances)

gap = (1.0 / (h.cr * (1.0 - gamma))) * distance_m1_m2

upper_bounds = np.array(v1, dtype=float)
for i in range(len(upper_bounds)):
    upper_bounds[i] += gap

# Display
print('Display environments')
m1.render()
m2.render()
print('Value function 1        :')
m1.display_to_m(v1)
print('Value function 2        :')
m1.display_to_m(v2)
print('Difference between both :')
m1.display_to_m(dv)
print()

print('Distance between states :\n', state_distances)
print('Distance between MDPs   : ', distance_m1_m2)
print('Predicted maximum gap   : ', gap)
print('Upper-bounds for MDP 2  :\n', upper_bounds)
