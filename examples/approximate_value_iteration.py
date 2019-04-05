import numpy as np

from llrl.envs.gridworld import GridWorld
from llrl.algorithms.value_iteration import approximate_value_iteration
from simple_rl.tasks.grid_world.GridWorldStateClass import GridWorldState


def example():
    size = 5
    gamma = .9
    epsilon = .1
    delta = .05
    fancy_plot = True

    # Create environment
    env = GridWorld(
        width=size, height=size, init_loc=(1, 1), goal_locs=[(size, size)],
        gamma=gamma, slip_prob=.5, goal_reward=1.0, is_goal_terminal=True
    )

    # Run approximate value iteration
    value_function = approximate_value_iteration(env, gamma, epsilon, delta)

    # Print computed value function
    print('Computed value function:')
    if fancy_plot:
        for j in range(size, 0, -1):
            for i in range(1, size + 1):
                print('{:>18}'.format(round(value_function[GridWorldState(i, j)], 2)), end=' ')
            print()
    else:
        for s in value_function:
            print('Value of', str(s), ':', value_function[s])


if __name__ == '__main__':
    np.random.seed(1993)
    example()
