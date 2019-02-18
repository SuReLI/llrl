from llrl.envs.gridworld import GridWorld
from llrl.algorithms.dynamic_programming import dynamic_programming
from llrl.envs.handler import Handler
import numpy as np


def print_state_distances(d):
    n = d.shape[0]
    assert n == d.shape[1]
    diag = np.zeros(shape=n)
    for i in range(n):
        diag[i] = d[i, i]
    print('distances :', diag)
    print('max       :', diag.max())


def transfer(v_source, match_matrix, lipschitz_constant, distance):
    upper_bound = np.dot(match_matrix, v_source) + lipschitz_constant * distance
    return upper_bound


def wasserstein_mdp_distance_test(m1, m2, v1, v2, state_distances_m1_m2, gamma, h):
    distance_m1_m2, match_m1_m2 = h.wasserstein_mdp_distance(m1, m2, state_distances_m1_m2)
    lipschitz_constant = (float(m1.nS) / (h.cr * (1.0 - gamma)))
    gap = lipschitz_constant * distance_m1_m2
    upper_bound = transfer(v2, match_m1_m2, lipschitz_constant, distance_m1_m2)

    print('\n\nTest wasserstein_mdp_distance')
    print('Display environments')
    m1.render()
    m2.render()
    print('Value function 1        :')
    m1.display_to_m(v1)
    print('Value function 2        :')
    m1.display_to_m(v2)
    print('Difference between both :')
    m1.display_to_m(abs(v1 - v2))
    print('Distance between states :\n', state_distances_m1_m2)
    print('Match m1, m2            :\n', match_m1_m2)
    print('Distance between MDPs d : ', distance_m1_m2)
    print('Lipschitz constant    K : ', lipschitz_constant)
    print('Gap               K * d : ', gap)
    print('Transferred upper-bound from m2 to m1:')
    m1.display_to_m(upper_bound)


def best_match_mdp_distance_test(m1, m2, v1, v2, state_distances_m1_m2, gamma, h):
    distance_m1_m2, match_m1_m2 = h.best_match_mdp_distance(m1, m2, state_distances_m1_m2)
    lipschitz_constant = 1.0 / (h.cr * (1.0 - gamma))
    upper_bound = transfer(v2, match_m1_m2, lipschitz_constant, distance_m1_m2)

    print('\n\nTest best_match_mdp_distance')
    print('Display environments')
    m1.render()
    m2.render()
    print('Value function 1        :')
    m1.display_to_m(v1)
    print('Value function 2        :')
    m1.display_to_m(v2)
    print('Difference between both :')
    m1.display_to_m(abs(v1 - v2))
    print('Distance between states :\n', state_distances_m1_m2)
    print('Match m1, m2            :\n', match_m1_m2)
    print('Lipschitz constant      :', lipschitz_constant)
    print('distance                :', distance_m1_m2)
    print('Transferred upper-bound from m2 to m1:')
    m1.display_to_m(upper_bound)


if __name__ == "__main__":
    np.set_printoptions(linewidth=200)

    # Parameters
    h = Handler()
    m1 = GridWorld(map_name='maze1', is_slippery=False)
    m2 = GridWorld(map_name='maze2', is_slippery=False)
    gamma = 0.9
    state_distances_m1_m2 = h.bi_simulation_distance(m1, m2, 0.1)
    v1 = dynamic_programming(m1, gamma=gamma, threshold=1e-10, iter_max=1000, verbose=False)
    v2 = dynamic_programming(m2, gamma=gamma, threshold=1e-10, iter_max=1000, verbose=False)

    wasserstein_mdp_distance_test(m1, m2, v1, v2, state_distances_m1_m2, gamma, h)
    best_match_mdp_distance_test(m1, m2, v1, v2, state_distances_m1_m2, gamma, h)
