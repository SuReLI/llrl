"""
Useful functions for making experiments (e.g. Lifelong RL)
"""

import time
from collections import defaultdict

from llrl.utils.utils import csv_write, mean_confidence_interval
from simple_rl.experiments import Experiment
from simple_rl.run_experiments import run_single_agent_on_mdp


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
    - Tasks are first sampled so that agents experience the same tasks;
    - Track and plot cumulative return for each task with confidence interval.

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
    :param resample_at_terminal: (bool)
    :param cumulative_plot: (bool)
    :param dir_for_plot: (str)
    :return:
    """
    if resample_at_terminal:
        print('Warning: not implemented in this tweaked version of run_agents_lifelong')

    # Experiment (for reproducibility, plotting).
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

    # Record how long each agent spends learning.
    print("Running experiment: \n" + str(experiment))
    start = time.clock()

    times = defaultdict(float)

    # Sample tasks at first so that agents experience the same tasks
    tasks = []
    for _ in range(samples):
        tasks.append(mdp_distr.sample())

    # Learn.
    for agent in agents:
        print(str(agent) + " is learning.")
        start = time.clock()

        cumulative_return_per_task = [0.] * samples
        cumulative_return_per_task_lo = [0.] * samples
        cumulative_return_per_task_up = [0.] * samples

        for i in range(samples):
            print("  Experience task " + str(i + 1) + " of " + str(samples) + ".")

            # Select the MDP.
            mdp = tasks[i]

            # Run the agent.
            hit_terminal, total_steps_taken, cumulative_returns = run_single_agent_on_mdp(
                agent, mdp, episodes, steps, experiment, verbose,
                track_disc_reward, reset_at_terminal, resample_at_terminal
            )

            cumulative_return_per_task[i],\
            cumulative_return_per_task_lo[i],\
            cumulative_return_per_task_up[i] =\
                mean_confidence_interval(cumulative_returns)

            # If we re-sample at terminal, keep grabbing MDPs until we're done.
            while resample_at_terminal and hit_terminal and total_steps_taken < steps:
                mdp = mdp_distr.sample()
                hit_terminal, steps_taken, _ = run_single_agent_on_mdp(
                    agent, mdp, episodes, steps - total_steps_taken, experiment, verbose,
                    track_disc_reward, reset_at_terminal, resample_at_terminal
                )
                total_steps_taken += steps_taken

            # Reset the agent.
            agent.reset()

        print(cumulative_return_per_task)  # TODO remove
        print(cumulative_return_per_task_lo)  # TODO remove
        print(cumulative_return_per_task_up)  # TODO remove

        # Track how much time this agent took.
        end = time.clock()
        times[agent] = round(end - start, 3)

    # Time stuff.
    print("\n--- TIMES ---")
    for agent in times.keys():
        print(str(agent) + " agent took " + str(round(times[agent], 2)) + " seconds.")
    print("-------------\n")

    experiment.make_plots(open_plot=open_plot)