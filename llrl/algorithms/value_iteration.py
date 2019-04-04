import numpy as np


def value_iteration(env, gamma=0.9, threshold=0.01, iter_max=1000, verbose=True):
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
    nS = env.number_of_states()
    nA = env.number_of_actions()
    value_function = np.zeros(shape=nS, dtype=float)
    convergence, niter = False, 0
    for i in range(iter_max):
        tmp = (-1e99) * np.ones(shape=value_function.shape, dtype=float)
        for s in range(nS):
            for a in range(nA):
                v_a = env.expected_reward(s, a) + gamma * np.dot(env.transition_probability_distribution(s, a), value_function)
                tmp[s] = max(v_a, tmp[s])
        delta = 0.0
        for s in range(nS):
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
    actions = env.actions()
    m_r = np.log(2. / delta) / (2. * epsilon**2)
    m_t = 2. * (np.log(2**(len(states)) - 2.) - np.log(delta)) / (epsilon**2)
    n_iter = int(max(m_r, m_t))

    print(m_r)
    print(m_t)
    print(n_iter)
    exit()

    #for s in states:
    #return reward_model, transition_model


def approximate_value_iteration(env, gamma, epsilon, verbose=True):
    """
    Approximate Value iteration algorithm.

    Required features from the environment:
    env.states() (array-like containing each state of the environment)
    env.actions() (array-like containing each action of the environment)
    env.reward_func(s, a) (sample a r(s, a))
    env.transition_func(s, a) (sample resulting state from application of a at s)

    :param env: environment object
    :param gamma: discount factor
    :param epsilon:
    :param verbose:
    :return: the value function as a dictionary
    """
    reward_model, transition_model = approximate_model(env)
