"""
Comparison of the bounds provided by R-MAX and Lipschitz-R-Max.

Setting:
- Using two grid-world MDPs with same transition functions and different reward functions while reaching goal;
- Learning on the first MDP and transferring the Lipschitz bound to the second one;
- Plotting percentage of bound use vs amount of prior knowledge + speed-up.
"""

import sys
import os
import numpy as np
import pandas as pd
from cycler import cycler
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib.ticker import MaxNLocator

from llrl.utils.utils import mean_confidence_interval
from llrl.utils.save import csv_write
from llrl.utils.chart_utils import color_ls
from llrl.utils.chart_utils import COLOR_SHIFT
from llrl.envs.gridworld import GridWorld
from llrl.agents.experimental.lrmax_prior_use import LRMaxExp
from simple_rl.run_experiments import run_single_agent_on_mdp

ROOT_PATH = 'results/prior_use/'

GAMMA = 0.9

N_INSTANCES = 3
N_EPISODES = 1000
N_STEPS = 1000

PRIOR_MIN = (1. + GAMMA) / (1. - GAMMA)
PRIOR_MAX = 0.
# PRIORS = [round(p, 1) for p in np.linspace(start=PRIOR_MIN, stop=PRIOR_MAX, num=10)]
PRIORS = [19.0, 15.0, 11.0, 10.0, 0.0]
PRIORS = [10.9, 10.6, 10.3]


def get_path_computation_number(agent_name):
    return ROOT_PATH + agent_name + 'computation_number.csv'


def get_path_time_step(agent_name):
    return ROOT_PATH + agent_name + 'time_step.csv'


def save_result(results, name):
    path = get_path_computation_number(name)
    csv_write(['prior_use_ratio_mean', 'prior_use_ratio_lo', 'prior_use_ratio_up'], path, mode='w')

    length = max([len(r) for r in results])
    for i in range(length):
        data_i = []
        for r in results:
            if len(r) > i:
                data_i.append(r[i][1])
        mean, lo, up = mean_confidence_interval(data_i)
        csv_write([mean, lo, up], path, mode='a')

    path = get_path_time_step(name)
    csv_write(['time_step', 'prior_use_ratio'], path, mode='w')
    for r in results:
        for row in r:
            csv_write([row[0], row[1]], path, mode='a')


def plot_time_step_results(names, open_plot=True):
    # LaTeX rendering
    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Set markers and colors
    markers = ['o', 's', 'D', '^', '*', 'x', 'p', '+', 'v', '|']
    colors = [[shade / 255.0 for shade in rgb] for rgb in color_ls]
    colors = colors[COLOR_SHIFT:] + colors[:COLOR_SHIFT]
    ax.set_prop_cycle(cycler('color', colors))

    for i in range(len(names)):
        df = pd.read_csv(get_path_time_step(names[i]))
        time_step = df.time_step
        prior_use_ratio = df.prior_use_ratio

        plt.scatter(time_step, prior_use_ratio, label=names[i], marker=markers[i])

    plt.xlim((0, 10000))
    plt.xlabel(r'Time Step')
    plt.ylabel(r'\% Prior Use')
    plt.legend(loc='best')
    plt.grid(True)
    # plt.title('')

    # Save
    plot_file_name = os.path.join(ROOT_PATH + 'prior_use_vs_time_step.pdf')
    plt.savefig(plot_file_name, format='pdf')

    # Open
    if open_plot:
        open_prefix = 'gnome-' if sys.platform == 'linux' or sys.platform == 'linux2' else ''
        os.system(open_prefix + 'open ' + plot_file_name)

    # Clear and close
    plt.cla()
    plt.close()


def plot_computation_number_results(names, open_plot=True):
    # LaTeX rendering
    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Set markers and colors
    markers = ['o', 's', 'D', '^', '*', 'x', 'p', '+', 'v', '|']
    colors = [[shade / 255.0 for shade in rgb] for rgb in color_ls]
    colors = colors[COLOR_SHIFT:] + colors[:COLOR_SHIFT]
    ax.set_prop_cycle(cycler('color', colors))

    for i in range(len(names)):
        df = pd.read_csv(get_path_computation_number(names[i]))
        prior_use_ratio_mean = df.prior_use_ratio_mean
        prior_use_ratio_lo = df.prior_use_ratio_lo
        prior_use_ratio_up = df.prior_use_ratio_up

        x = range(len(prior_use_ratio_mean))

        plt.plot(x, prior_use_ratio_mean, '-o', label=names[i], marker=markers[i])
        plt.fill_between(x, prior_use_ratio_lo, prior_use_ratio_up, alpha=.25, facecolor=colors[i], edgecolor=colors[i])

    plt.ylim((-6., 106.))
    plt.xlabel(r'Computation Number')
    plt.ylabel(r'\% Prior Use')
    plt.legend(loc='best')
    plt.grid(True)
    # plt.title('')

    # Save
    plot_file_name = os.path.join(ROOT_PATH + 'prior_use_vs_computation_number.pdf')
    plt.savefig(plot_file_name, format='pdf')

    # Open
    if open_plot:
        open_prefix = 'gnome-' if sys.platform == 'linux' or sys.platform == 'linux2' else ''
        os.system(open_prefix + 'open ' + plot_file_name)

    # Clear and close
    plt.cla()
    plt.close()


def prior_use_experiment(run_experiment=True, open_plot=True, verbose=True):
    """
    Prior use experiment:
    Record the ratio of prior use during the model's distance computation in the simple setting of interacting
    sequentially with two different environments.
    :param run_experiment: set to False for plot only
    :param open_plot: set to False to disable plot (only saving)
    :return: None
    """
    w = 4
    h = 3
    env1 = GridWorld(
        width=w, height=h, init_loc=(2, 1), goal_locs=[(w, h)],
        slip_prob=0.1, goal_reward=0.9, is_goal_terminal=False
    )
    env2 = GridWorld(
        width=w, height=h, init_loc=(2, 1), goal_locs=[(w, h)],
        slip_prob=0.2, goal_reward=1.0, is_goal_terminal=False
    )

    # Compute needed number of samples for L-R-MAX to achieve epsilon optimal behavior with probability (1 - delta)
    epsilon = .1
    delta = .05
    m_r = np.log(2. / delta) / (2. * epsilon ** 2)
    m_t = 2. * (np.log(2 ** (float(w * h)) - 2.) - np.log(delta)) / (epsilon ** 2)
    m = int(max(m_r, m_t))

    names = []

    for p in PRIORS:
        results = []
        name = 'default'
        for i in range(N_INSTANCES):
            agent = LRMaxExp(
                actions=env1.get_actions(),
                gamma=GAMMA,
                count_threshold=m,
                epsilon=epsilon,
                prior=p
            )
            name = agent.name

            if run_experiment:
                if verbose:
                    print('Running instance', i + 1, 'of', N_INSTANCES, 'for agent', name)

                run_single_agent_on_mdp(
                    agent, env1, episodes=N_EPISODES, steps=N_STEPS, experiment=None, verbose=False,
                    track_disc_reward=False, reset_at_terminal=False, resample_at_terminal=False
                )
                agent.reset()
                run_single_agent_on_mdp(
                    agent, env2, episodes=N_EPISODES, steps=N_STEPS, experiment=None, verbose=False,
                    track_disc_reward=False, reset_at_terminal=False, resample_at_terminal=False
                )

                results.append(agent.get_results())

        names.append(name)

        # Save results
        if run_experiment:
            save_result(results, name)

    # Plot
    plot_computation_number_results(names, open_plot)
    plot_time_step_results(names, open_plot)


if __name__ == '__main__':
    np.random.seed(1993)
    prior_use_experiment(run_experiment=True, open_plot=True, verbose=True)
