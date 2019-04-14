import numpy as np

from llrl.envs.gridworld import GridWorld
from llrl.algorithms.value_iteration import approximate_value_iteration
from simple_rl.tasks.grid_world.GridWorldStateClass import GridWorldState


def sample_maze(size, gamma):
    w, h = size, size

    walls = [
        (2, 2), (3, 2), (4, 2), (5, 2), (6, 2), (4, 3),
        (2, 4), (2, 5), (2, 6), (4, 5), (5, 5), (4, 4)
    ]
    sampled_slip_prob = 0.1  # np.random.uniform(0., 0.1)

    env = GridWorld(
        width=w, height=h, init_loc=(1, 1), rand_init=False, goal_locs=[(w, h)], lava_locs=[()], walls=walls,
        is_goal_terminal=True, gamma=gamma, slip_prob=sampled_slip_prob, step_cost=0.0, lava_cost=0.01,
        goal_reward=1, name="Maze"
    )

    return env


def example():
    size = 6
    gamma = .9
    epsilon = .1
    delta = .05
    fancy_plot = True

    # Create environment
    env = sample_maze(size, gamma)

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
