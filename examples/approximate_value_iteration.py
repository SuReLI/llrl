import numpy as np

from llrl.envs.gridworld import GridWorld
from llrl.algorithms.value_iteration import approximate_value_iteration


def example():
    size = 5
    gamma = .9
    epsilon = .1
    delta = .05

    # Create environment
    env = GridWorld(
        width=size, height=size, init_loc=(1, 1), goal_locs=[(size, size)],
        gamma=gamma, slip_prob=0.0, goal_reward=1.0, name="grid-world"
    )

    # Run approximate value iteration
    value_function = approximate_value_iteration(env, gamma, epsilon, delta)

    # Print computed value function
    for s in value_function:
        print('V(', str(s), ') =', value_function[s])


if __name__ == '__main__':
    np.random.seed(1993)
    example()
