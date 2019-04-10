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
from simple_rl.run_experiments import run_agents_on_mdp

GAMMA = 0.9

N_EPISODES = 100
N_STEPS = 1000

PRIOR_MIN = (1. + GAMMA) / (1. - GAMMA)
PRIOR_MAX = 0.
PRIORS = [round(p, 1) for p in np.linspace(start=PRIOR_MIN, stop=PRIOR_MAX, num=3)]


def get_path(agent):
    return 'results/prior_use/' + agent.name + '.csv'


def save_results(agents):
    for agent in agents:
        path = 'results/prior_use/' + agent.name + '.csv'
        csv_write(agent.prior_use_counter[0], path, mode='w')
        for i in range(1, len(agent.prior_use_counter) - 1):
            row = agent.prior_use_counter[i]
            csv_write(row, path, mode='a')


def plot_results(agents, open_plot=True):
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

    for i in range(len(agents)):
        df = pd.read_csv(get_path(agents[i]))
        n_computation = df.n_computation
        n_prior_use = df.n_prior_use
        ratio = round(100. * n_prior_use / n_computation, 2)

        plt.plot(range(len(ratio)), ratio, '-o', label=agents[i].name, marker=markers[i])

    plt.xlabel(r'Computation Number')
    plt.ylabel(r'\% Prior Use')
    plt.legend(loc='best')
    plt.grid(True)
    # plt.title('')

    # Save
    plot_file_name = os.path.join('results/prior_use/prior_use.pdf')
    plt.savefig(plot_file_name, format='pdf')

    # Open
    if open_plot:
        open_prefix = 'gnome-' if sys.platform == 'linux' or sys.platform == 'linux2' else ''
        os.system(open_prefix + 'open ' + plot_file_name)

    # Clear and close
    plt.cla()
    plt.close()


def prior_use_experiment(only_plot=False, open_plot=True):
    size = 2
    env1 = GridWorld(width=size, height=size, init_loc=(1, 1), goal_locs=[(size, size)], slip_prob=0.1, goal_reward=0.9)
    env2 = GridWorld(width=size, height=size, init_loc=(1, 1), goal_locs=[(size, size)], slip_prob=0.2, goal_reward=1.0)

    # Compute needed number of samples for L-R-MAX to achieve epsilon optimal behavior with probability (1 - delta)
    epsilon = .1
    delta = .05
    m_r = np.log(2. / delta) / (2. * epsilon ** 2)
    m_t = 2. * (np.log(2 ** (float(size * size)) - 2.) - np.log(delta)) / (epsilon ** 2)
    m = int(max(m_r, m_t))

    agents = []

    for p in PRIORS:
        agent = LRMaxExp(
            actions=env1.get_actions(),
            gamma=GAMMA,
            count_threshold=m,
            epsilon=epsilon,
            prior=p
        )
        agents.append(agent)

        if not only_plot:
            run_agents_on_mdp([agent], env1, instances=1, episodes=N_EPISODES, steps=N_STEPS,
                              reset_at_terminal=True, verbose=False, open_plot=False)
            run_agents_on_mdp([agent], env2, instances=1, episodes=N_EPISODES, steps=N_STEPS,
                              reset_at_terminal=True, verbose=False, open_plot=False)

    # Save results
    if not only_plot:
        save_results(agents)

    # Plot
    plot_results(agents, open_plot)


if __name__ == '__main__':
    np.random.seed(1993)
    prior_use_experiment(True)
