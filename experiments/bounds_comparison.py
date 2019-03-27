"""
Comparison of the bounds provided by R-MAX and Lipschitz-R-Max.

Setting:
- Using two grid-world MDPs with same transition functions and different reward functions while reaching goal;
- Learning on the first MDP and transferring the Lipschitz bound to the second one;
- Plotting percentage of bound use vs amount of prior knowledge.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc

from llrl.utils.utils import csv_write
from llrl.envs.gridworld import GridWorld
from llrl.agents.lrmax_constant_transitions_testing import LRMaxCTTesting
from simple_rl.run_experiments import run_agents_on_mdp


SAVE_PATH = "results/bounds_comparison_results.csv"
ENTRIES = ["delta_r", "ratio_rmax_bound_use", "ratio_lip_bound_use", "n_time_steps", "n_time_steps_cv"]
# PRIOR = np.linspace(0., 1., num=21)
# PRIOR = [1., .9, .8, .7, .6, .5, .3, .28, .26, .24, .22, .2, .1, .0]
PRIOR = [.0, .25, .5, .75, 1.]


def plot_results():
    df = pd.read_csv(SAVE_PATH)
    prior = df.delta_r
    perct = 100. * df.ratio_lip_bound_use

    # LaTeX rendering
    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plt.plot(prior, perct, '-o')
    plt.xlim(1., 0.)  # decreasing upper-bound
    plt.xlabel(r'Prior knowledge: known upper-bound on $\max_{s, a} = |R_s^a - \bar{R}_s^a|$')
    plt.ylabel(r'\% use Lipschitz bound')
    # plt.title('')
    plt.grid(True, linestyle='--')
    plt.show()


def bounds_test():
    # MDP
    size = 2
    mdp1 = GridWorld(width=size, height=size, init_loc=(1, 1), goal_locs=[(size, size)], goal_reward=1.0)
    mdp2 = GridWorld(width=size, height=size, init_loc=(1, 1), goal_locs=[(size, size)], goal_reward=0.8)

    csv_write(ENTRIES, SAVE_PATH, 'w')

    for prior in PRIOR:
        lrmaxct = LRMaxCTTesting(actions=mdp1.get_actions(), gamma=.9, count_threshold=1, delta_r=prior, path=SAVE_PATH)

        # Run twice
        run_agents_on_mdp([lrmaxct], mdp1, instances=1, episodes=100, steps=30,
                          reset_at_terminal=True, verbose=False, open_plot=False)

        run_agents_on_mdp([lrmaxct], mdp2, instances=1, episodes=100, steps=30,
                          reset_at_terminal=True, verbose=False, open_plot=False)


def main():
    bounds_test()
    plot_results()


if __name__ == "__main__":
    main()
