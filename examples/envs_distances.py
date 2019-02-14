from llrl.envs.gridworld import GridWorld
from llrl.envs.handler import Handler

envs_names = ['maze1', 'maze2']
envs = [GridWorld(map_name=n, is_slippery=False) for n in envs_names]

h = Handler()

d = h.bi_simulation_distance(envs[0], envs[1])
dist = h.wasserstein_mdp_distance(envs[0], envs[1], d)

print('Render environments:')
envs[0].render()
envs[1].render()
print('States distance matrix:\n', d)
print('MDP distance:\n', dist)