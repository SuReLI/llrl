import copy
import numpy as np
from collections import defaultdict


def value_iteration(env, gamma=.9, threshold=.01, iter_max=1000, verbose=True):
    """
    Exact Value iteration algorithm.

    Required features from the environment:
    env.number_of_states()
    env.number_of_actions()
    env.expected_reward(s, a)
    env.transition_probability_distribution(s, a)

    Expect the states of the environment to be integers.

    :param env: environment object
    :param gamma: discount factor
    :param threshold: threshold for convergence (algorithms stops when the infinite
    norm between subsequent value functions is below @threshold)
    :param iter_max: upper bound on the maximum number of iterations
    :param verbose:
    :return: value function as a vector
    """
    n_states = env.number_of_states()
    n_actions = env.number_of_actions()
    value_function = np.zeros(shape=n_states, dtype=float)
    convergence, niter = False, 0
    for i in range(iter_max):
        tmp = (-1e99) * np.ones(shape=value_function.shape, dtype=float)
        for s in range(n_states):
            for a in range(n_actions):
                v_a = env.expected_reward(s, a) +\
                      gamma * np.dot(env.transition_probability_distribution(s, a), value_function)
                tmp[s] = max(v_a, tmp[s])
        delta = 0.0
        for s in range(n_states):
            delta = max(delta, abs(value_function[s] - tmp[s]))
        if verbose:
            print('Iteration', i+1, '/', iter_max, 'maximum difference =', delta)
        value_function = np.array(tmp)
        if delta < threshold:
            convergence, niter = True, i+1
            break
    if verbose:
        if convergence:
            print('Convergence after', niter, 'iterations.')
        else:
            print('Maximum number of iteration (', iter_max, ') reached before convergence.')
            print('Tolerance was', threshold)
    return value_function


def approximate_model(env, epsilon=.1, delta=.05):
    """
    Compute an approximate model for the input environment.

    :param env: see approximate_value_iteration function
    :param epsilon: precision of the model (1-norm)
    :param delta: the model is epsilon-accurate with probability 1 - delta
    :return: tuple(dictionary, dictionary) reward_model, transition_model
    """
    states = env.states()
    actions = env.actions
    m_r = np.log(2. / delta) / (2. * epsilon**2)
    m_t = 2. * (np.log(2**(len(states)) - 2.) - np.log(delta)) / (epsilon**2)
    n_samples = int(max(m_r, m_t))

    sampled_rewards = defaultdict(lambda: defaultdict(list))
    sampled_transitions = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for s in states:
        for a in actions:
            for i in range(n_samples):
                r = env.reward_func(s, a)
                s_p = env.transition_func(s, a)
                sampled_rewards[s][a] += [r]
                sampled_transitions[s][a][s_p] += 1

    reward_model = defaultdict(lambda: defaultdict(float))
    transition_model = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

    for s in states:
        for a in actions:
            reward_model[s][a] = sum(sampled_rewards[s][a]) / float(n_samples)
            for s_p in sampled_transitions[s][a]:
                transition_model[s][a][s_p] = sampled_transitions[s][a][s_p] / float(n_samples)

    return reward_model, transition_model


def approximate_value_iteration(env, gamma=.9, epsilon=.1, delta=.05, verbose=True):
    """
    Approximate Value iteration algorithm.

    Required features from the environment:
    env.states() (array-like containing each state of the environment)
    env.actions (array-like containing each action of the environment)
    env.reward_func(s, a) (sample a r(s, a))
    env.transition_func(s, a) (sample resulting state from application of a at s)

    :param env: environment object
    :param gamma: discount factor
    :param epsilon: precision of the model (1-norm) and precision of the computed value function
    :param delta: the model is epsilon-accurate with probability 1 - delta
    :param verbose:
    :return: the value function as a dictionary
    """
    if verbose:
        print('Running approximate value iteration.')
        print('gamma   :', gamma)
        print('epsilon :', epsilon)
        print('delta   :', delta)
    states = env.states()
    actions = env.actions
    reward_model, transition_model = approximate_model(env, epsilon, delta)

    value_function = defaultdict(float)
    # for s in states:
    #     value_function[s] = 0.

    n_iter = int(np.log(1. / (epsilon * (1. - gamma))) / (1. - gamma))  # Nb of value iterations

    for i in range(n_iter):
        if verbose:
            print('Iteration', i, '/', n_iter)
        tmp = copy.deepcopy(value_function)
        for s in states:
            v_s = 0.
            for a in actions:
                v_s_p = 0.
                for s_p in transition_model[s][a]:
                    v_s_p = transition_model[s][a][s_p] * value_function[s_p]
                q_sa = reward_model[s][a] + gamma * v_s_p
                if v_s < q_sa:
                    v_s = q_sa
            tmp[s] = v_s
        value_function = copy.deepcopy(tmp)

    return value_function
