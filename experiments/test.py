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
lipschitz_constant = (1.0 / (h.cr * (1.0 - gamma)))
gap = lipschitz_constant * distance_m1_m2

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
print('Distance between MDPs d = ', distance_m1_m2)
print('Lipschitz constant    K = ', lipschitz_constant)
print('Gap               K * d = ', gap)
print('Upper-bounds for MDP 2  :')
m1.display_to_m(upper_bounds)
