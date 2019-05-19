"""
Negative transfer example using Song et al. [2016] transfer method.
"""


import numpy as np


import llrl.algorithms.mdp_distances as dst
from llrl.algorithms.value_iteration import value_iteration
from llrl.envs.n_states import NStates


GAMMA = 0.9
THRESHOLD = 1e-10


def q_function_from_value_function(env, v):
    ns = env.number_of_states()
    na = env.number_of_actions()
    q = np.zeros(shape=(ns, na))
    for s in range(ns):
        for a in range(na):
            v_next = 0.
            for s_p in range(ns):
                v_next += env.T[s][a][s_p] * v[s_p]
            q[s][a] = env.expected_reward(s, a) + GAMMA * v_next
    return q


def weighted_transfer(v_source, q_source, matching_matrix):
    v = np.zeros(v_source.shape)
    q = np.zeros(q_source.shape)
    ns = len(v)
    for i in range(ns):
        for j in range(ns):
            v[i] += matching_matrix[i][j] * v_source[j] / float(ns)
            for a in range(2):
                q[i][a] += matching_matrix[i][j] * q_source[j][a] / float(ns)
    return v, q


def weighted_transfer_test():
    r1 = [
        [1., 0.],
        [0., 0.]
    ]
    r2 = [
        [0., 0.],
        [0., 1.]
    ]
    ns = len(r1)
    env1 = NStates(ns, r1)
    env2 = NStates(ns, r2)

    states_distances = dst.bi_simulation_distance(env1, env2)
    environments_distances, matching_matrix = dst.wasserstein_mdp_distance(env1, env2, states_distances)

    v1_true = value_iteration(env1, gamma=GAMMA, threshold=THRESHOLD, verbose=False)
    q1_true = q_function_from_value_function(env1, v1_true)

    v2_true = value_iteration(env2, gamma=GAMMA, threshold=THRESHOLD, verbose=False)
    q2_true = q_function_from_value_function(env2, v2_true)

    v2_transfer, q2_transfer = weighted_transfer(v1_true, q1_true, matching_matrix)

    print('Weighted transfer')
    print('\nM1 True value function    :\n', v1_true)
    print('\nM1 True Q-function        :\n', q1_true)
    print('\nM2 True value function    :\n', v2_true)
    print('\nM2 True Q-function        :\n', q2_true)
    print('\nM2 Transferred Q-function :\n', q2_transfer)


def state_transfer_test():
    print('TODO: state transfer')


if __name__ == "__main__":
    weighted_transfer_test()
    # state_transfer_test()
