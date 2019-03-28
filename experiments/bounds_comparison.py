"""
Comparison of the bounds provided by R-MAX and Lipschitz-R-Max.

Setting:
- Using two grid-world MDPs with same transition functions and different reward functions while reaching goal;
- Learning on the first MDP and transferring the Lipschitz bound to the second one;
- Plotting percentage of bound use vs amount of prior knowledge.
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc

from llrl.utils.utils import csv_write, mean_confidence_interval
from llrl.envs.gridworld import GridWorld
from llrl.agents.lrmax_ct_exp import LRMaxCTExp
from llrl.agents.rmax_vi_exp import RMaxVIExp
from simple_rl.run_experiments import run_agents_on_mdp


SAVE_PATH = 'results/bounds_comparison_results.csv'
LRMAX_TMP_SAVE_PATH = 'results/tmp/bounds_comparison_results_LRMAX.csv'
RMAX_TMP_SAVE_PATH = 'results/tmp/bounds_comparison_results_RMAX.csv'

N_INSTANCES = 100

# PRIOR = [1., .9, .8, .7, .6, .5, .4, .35, .3, .25, .2, .1, .0]
PRIOR = [.0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.]


def plot_results(path):
    # 1. Results in terms of bound-use
    df = pd.read_csv(path)
    prior = df.prior
    ratio = df.ratio_lip_bound_use_mean
    ratio_up = df.ratio_lip_bound_use_upper
    ratio_lo = df.ratio_lip_bound_use_lower
    speed_up = df.speed_up_mean
    speed_up_up = df.speed_up_upper
    speed_up_lo = df.speed_up_lower

    # LaTeX rendering
    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    colors = ['steelblue', 'darkorange', 'green']
    plt.plot(prior, ratio, '-o', color=colors[0], label=r'\% use Lipschitz bound')
    plt.fill_between(prior, ratio_up, ratio_lo, color=colors[0], alpha=0.2)
    plt.plot(prior, speed_up, '-^', color=colors[1], label=r'\% time-steps gained (speed-up)')
    plt.fill_between(prior, speed_up_up, speed_up_lo, color=colors[1], alpha=0.2)

    plt.xlim(1., 0.)  # decreasing upper-bound
    plt.xlabel(r'Prior knowledge (known upper-bound on $\max_{s, a} = |R_s^a - \bar{R}_s^a|$)')
    plt.ylabel(r'\%')
    plt.legend(loc='best')
    # plt.title('')
    plt.grid(True, linestyle='--')
    plt.show()


def bounds_test(verbose=False):
    # MDP
    sz = 3
    mdp1 = GridWorld(width=sz, height=sz, init_loc=(1, 1), goal_locs=[(sz, sz)], goal_reward=1.0)
    mdp2 = GridWorld(width=sz, height=sz, init_loc=(1, 1), goal_locs=[(sz, sz)], goal_reward=0.8)

    results = []

    for _ in range(N_INSTANCES):
        csv_write(['prior', 'ratio_rmax_bound_use', 'ratio_lip_bound_use', 'lrmax_n_time_steps', 'lrmax_n_time_steps_cv'], LRMAX_TMP_SAVE_PATH, 'w')
        csv_write(['rmax_n_time_steps', 'rmax_n_time_steps_cv'], RMAX_TMP_SAVE_PATH, 'w')

        for prior in PRIOR:
            lrmaxct = LRMaxCTExp(actions=mdp1.get_actions(), gamma=.9, count_threshold=1, delta_r=prior, path=LRMAX_TMP_SAVE_PATH)
            rmaxvi = RMaxVIExp(actions=mdp1.get_actions(), gamma=.9, count_threshold=1, path=RMAX_TMP_SAVE_PATH)

            # Run twice
            run_agents_on_mdp([lrmaxct], mdp1, instances=1, episodes=100, steps=30,
                              reset_at_terminal=True, verbose=False, open_plot=False)

            run_agents_on_mdp([lrmaxct, rmaxvi], mdp2, instances=1, episodes=100, steps=30,
                              reset_at_terminal=True, verbose=False, open_plot=False)

        # Retrieve data
        df_lrmax = pd.read_csv(LRMAX_TMP_SAVE_PATH)
        df_rmax = pd.read_csv(RMAX_TMP_SAVE_PATH)
        df = pd.concat([df_lrmax, df_rmax], axis=1, sort=False)
        result = []
        for index, row in df.iterrows():
            prior = row['prior']
            ratio_lip_bound_use = 100. * row['ratio_lip_bound_use']
            speed_up = 100. * (row['rmax_n_time_steps_cv'] - row['lrmax_n_time_steps_cv']) / row['rmax_n_time_steps_cv']
            result.append([prior, ratio_lip_bound_use, speed_up])
        results.append(result)

    # Gather results
    csv_write(
        ['prior',
         'ratio_lip_bound_use_mean',
         'ratio_lip_bound_use_upper',
         'ratio_lip_bound_use_lower',
         'speed_up_mean',
         'speed_up_upper',
         'speed_up_lower'],
        SAVE_PATH, 'w'
    )
    for i in range(len(PRIOR)):
        rlbu = []
        su = []
        for result in results:
            rlbu.append(result[i][1])
            su.append(result[i][2])
        rlbu_mci = mean_confidence_interval(rlbu)
        su_mci = mean_confidence_interval(su)
        csv_write([PRIOR[i]] + list(rlbu_mci) + list(su_mci), SAVE_PATH, 'a')
    if verbose:
        for result in results:
            print(result)


if __name__ == '__main__':
    # bounds_test()
    # plot_results(SAVE_PATH)
    plot_results('results/bounds-use-and-speed-up/gridworld_h-2_w-2/data.csv')
