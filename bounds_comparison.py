"""
Comparison of the bounds provided by R-MAX and Lipschitz-R-Max.

Setting:
- Using two grid-world MDPs with same transition functions and different reward functions while reaching goal;
- Learning on the first MDP and transferring the Lipschitz bound to the second one;
- Plotting percentage of bound use vs amount of prior knowledge + speed-up.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
from tqdm import trange

from llrl.utils.env_handler import make_env_distribution
from llrl.utils.utils import mean_confidence_interval
from llrl.utils.save import csv_write
from llrl.envs.gridworld import GridWorld
from llrl.agents.experimental.lrmax_bounds_use import ExpLRMax
from llrl.agents.experimental.rmax_bounds_use import ExpRMax
from llrl.experiments import run_agents_on_mdp


def plot_bound_use(path):
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

    plt.xlim(max(prior), min(prior))  # decreasing upper-bound
    plt.xlabel(r'Prior knowledge (known upper-bound on $\max_{s, a} = D^{M \bar{M}}_{\gamma V^*_{\bar{M}}}(s, a)$)')
    plt.ylabel(r'\%')
    plt.legend(loc='best')
    # plt.title('')
    plt.grid(True, linestyle='-')
    plt.show()


def bounds_comparison_experiment(verbose=False, plot=True):
    # Parameters
    gamma = 0.9
    n_instances = 10
    n_episodes = 10  # 100
    n_steps = 10  # 30
    prior_min = 1.  # (1. + gamma) / (1. - gamma)
    prior_max = 0.
    priors = [round(p, 1) for p in np.linspace(start=prior_min, stop=prior_max, num=10)]
    save_path = 'results/bounds_comparison_results.csv'

    # Environments
    sz = 2
    gl = [(sz, sz)]
    n_states = int(sz * sz)
    mdp1 = GridWorld(width=sz, height=sz, init_loc=(1, 1), goal_locs=gl, goal_rewards=[0.8])
    mdp2 = GridWorld(width=sz, height=sz, init_loc=(1, 1), goal_locs=gl, goal_rewards=[1.0])

    env_distribution = make_env_distribution(env_class='tight', n_env=2, gamma=gamma, env_name='tight', version=2,
                                             w=11, h=11, stochastic=True, verbose=False)
    mdps = env_distribution.get_all_mdps()
    mdp1 = mdps[0]
    mdp2 = mdps[1]

    actions = mdp1.get_actions()
    r_max = 1.
    v_max = None
    deduce_v_max = True  # erase previous definition of v_max
    n_known = 1
    epsilon_q = .01
    epsilon_m = .01
    delta = .1

    results = []

    for _ in trange(n_instances, desc='{:>10}'.format('instances')):
        lrmax, rmax, df_lrmax, df_rmax = None, None, None, None
        for i in trange(len(priors), desc='{:>10}'.format('priors')):
            lrmax = ExpLRMax(actions=actions, gamma=gamma, r_max=r_max, v_max=v_max, deduce_v_max=deduce_v_max,
                             n_known=n_known, deduce_n_known=False, epsilon_q=epsilon_q, epsilon_m=epsilon_m,
                             delta=delta, n_states=n_states, max_memory_size=None, prior=priors[i],
                             estimate_distances_online=True, min_sampling_probability=.5, name="ExpLRMax")
            rmax = ExpRMax(actions=actions, gamma=gamma, r_max=r_max, v_max=v_max, deduce_v_max=deduce_v_max,
                           n_known=n_known, deduce_n_known=False, epsilon_q=epsilon_q, epsilon_m=epsilon_m, delta=delta,
                           n_states=n_states, name="ExpRMax")

            if i > 0:
                lrmax.data = df_lrmax
                rmax.data = df_rmax

            # Run twice
            lrmax.write_data = False
            run_agents_on_mdp([lrmax], mdp1, n_instances=1, n_episodes=n_episodes, n_steps=n_steps,
                              reset_at_terminal=False, verbose=False)

            lrmax.write_data = True
            run_agents_on_mdp([lrmax, rmax], mdp2, n_instances=1, n_episodes=n_episodes, n_steps=n_steps,
                              reset_at_terminal=False, verbose=False)

            # Retrieve data
            df_lrmax = lrmax.data
            df_rmax = rmax.data

        instance_result = []
        for i, _ in df_lrmax.iterrows():
            prior = df_lrmax['prior'][i]
            ratio_lip_bound_use = 100. * df_lrmax['ratio_lip_bound_use'][i]
            speed_up = 100. * (df_rmax['cnt_time_steps_cv'][i] - df_lrmax['cnt_time_steps_cv'][i]) / df_rmax['cnt_time_steps_cv'][i]
            instance_result.append([prior, ratio_lip_bound_use, speed_up])
        results.append(instance_result)

    # Gather results
    csv_write(['prior', 'ratio_lip_bound_use_mean', 'ratio_lip_bound_use_upper', 'ratio_lip_bound_use_lower',
               'speed_up_mean', 'speed_up_upper', 'speed_up_lower'], save_path, 'w')
    for i in range(len(priors)):
        rlbu = []
        su = []
        for result in results:
            rlbu.append(result[i][1])
            su.append(result[i][2])
        rlbu_mci = mean_confidence_interval(rlbu, confidence=.9)
        su_mci = mean_confidence_interval(su, confidence=.9)
        csv_write([priors[i]] + list(rlbu_mci) + list(su_mci), save_path, 'a')
    if verbose:
        for result in results:
            print(result)

    if plot:
        plot_bound_use(save_path)


if __name__ == '__main__':
    bounds_comparison_experiment(plot=True)
    # plot_bound_use('results/bounds-use-and-speed-up/gridworld_h-2_w-2/data.csv')
