"""
Useful functions for making experiments (e.g. Lifelong RL)
"""

import sys
import os
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import rc
from collections import defaultdict

from llrl.utils.utils import csv_write, mean_confidence_interval
from simple_rl.experiments import Experiment
from simple_rl.run_experiments import run_single_agent_on_mdp
from simple_rl.utils.chart_utils import _format_title


def plot_returns_vs_tasks(path, agents, returns_per_agent, open_plot=True):
    n_tasks = len(returns_per_agent[0])
    x = range(1, n_tasks + 1)

    # LaTeX rendering
    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    for i in range(len(agents)):
        mean_return = [returns_per_agent[i][j][0] for j in range(n_tasks)]
        mean_return_lo = [returns_per_agent[i][j][1] for j in range(n_tasks)]
        mean_return_up = [returns_per_agent[i][j][2] for j in range(n_tasks)]
        plt.plot(x, mean_return, '-o', label=agents[i])
        plt.fill_between(x, mean_return_lo, mean_return_up, alpha=0.2)

    plt.xlabel(r'Task number')
    plt.ylabel(r'Discounted return')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--')
    exp_dir_split_list = path.split("/")
    if 'results' in exp_dir_split_list:
        exp_name = exp_dir_split_list[exp_dir_split_list.index('results') + 1]
    else:
        exp_name = exp_dir_split_list[0]
    plt_title = _format_title('Discounted return: ' + exp_name)
    plt.title(plt_title)

    # Save
    plot_file_name = os.path.join(path, "returns_vs_tasks.pdf")
    plt.savefig(plot_file_name, format="pdf")

    # Open
    if open_plot:
        open_prefix = "gnome-" if sys.platform == "linux" or sys.platform == "linux2" else ""
        os.system(open_prefix + "open " + plot_file_name)

    # Clear and close
    plt.cla()
    plt.close()


def run_agents_lifelong(
        agents,
        mdp_distr,
        samples=5,
        episodes=1,
        steps=100,
        clear_old_results=True,
        open_plot=True,
        verbose=False,
        track_disc_reward=False,
        reset_at_terminal=False,
        resample_at_terminal=False,
        cumulative_plot=True,
        dir_for_plot="results"
):
    """
    Tweaked version of simple_rl.run_experiments.run_agents_lifelong
    Modifications are the following:
    - Tasks are first sampled so that agents experience the same sequence of tasks;
    - Track and plot return for each task with confidence interval.

    Runs each agent on the MDP distribution according to the given parameters.
    If @mdp_distr has a non-zero horizon, then gamma is set to 1 and @steps is ignored.

    :param agents: (list)
    :param mdp_distr: (MDPDistribution)
    :param samples: (int)
    :param episodes: (int)
    :param steps: (int)
    :param clear_old_results: (bool)
    :param open_plot: (bool)
    :param verbose: (bool)
    :param track_disc_reward: (bool) If true records and plots discounted reward, discounted over episodes.
    So, if each episode is 100 steps, then episode 2 will start discounting as though it's step 101.
    :param reset_at_terminal: (bool)
    :param resample_at_terminal: (bool) (not implemented in this tweaked version of run_agents_lifelong)
    :param cumulative_plot: (bool)
    :param dir_for_plot: (str)
    :return:
    """
    if resample_at_terminal:
        print('Warning: not implemented in this tweaked version of run_agents_lifelong')

    # Experiment (for reproducibility, plotting)
    exp_params = {"samples":samples, "episodes":episodes, "steps":steps}
    experiment = Experiment(
        agents=agents,
        mdp=mdp_distr,
        params=exp_params,
        is_episodic=episodes > 1,
        is_lifelong=True,
        clear_old_results=clear_old_results,
        track_disc_reward=track_disc_reward,
        cumulative_plot=cumulative_plot,
        dir_for_plot=dir_for_plot
    )

    print("Running experiment: \n" + str(experiment))
    times = defaultdict(float)
    returns_per_agent = []

    # Sample tasks at first so that agents experience the same sequence of tasks
    tasks = []
    for _ in range(samples):
        tasks.append(mdp_distr.sample())

    for agent in agents:
        print(str(agent) + " is learning.")
        start = time.clock()

        return_per_task = [(0., 0., 0.)] * samples  # Mean, lower confidence interval bound, upper

        for i in range(samples):
            print("  Experience task " + str(i + 1) + " of " + str(samples) + ".")

            # Select the MDP
            mdp = tasks[i]

            # Run the agent
            hit_terminal, total_steps_taken, return_per_episode = run_single_agent_on_mdp(
                agent, mdp, episodes, steps, experiment, verbose=verbose, track_disc_reward=track_disc_reward,
                reset_at_terminal=reset_at_terminal, resample_at_terminal=resample_at_terminal
            )

            return_per_task[i] = mean_confidence_interval(return_per_episode)

            # If we re-sample at terminal, keep grabbing MDPs until we're done
            while resample_at_terminal and hit_terminal and total_steps_taken < steps:
                mdp = mdp_distr.sample()
                hit_terminal, steps_taken, _ = run_single_agent_on_mdp(
                    agent, mdp, episodes, steps - total_steps_taken, experiment, verbose,
                    track_disc_reward, reset_at_terminal, resample_at_terminal
                )
                total_steps_taken += steps_taken

            # Reset the agent
            agent.reset()
        returns_per_agent.append(return_per_task)

        # Track how much time this agent took
        end = time.clock()
        times[agent] = round(end - start, 3)

    # Time stuff
    print("\n--- TIMES ---")
    for agent in times.keys():
        print(str(agent) + " agent took " + str(round(times[agent], 2)) + " seconds.")
    print("-------------\n")

    # Plot
    plot_returns_vs_tasks(experiment.exp_directory, agents, returns_per_agent)
    experiment.make_plots(open_plot=open_plot)