"""
Useful functions for making experiments (e.g. Lifelong RL)
"""

import time
from collections import defaultdict

from llrl.utils.utils import mean_confidence_interval
from llrl.utils.save import save
from llrl.utils.chart_utils import plot
from simple_rl.experiments import Experiment
from simple_rl.run_experiments import run_single_agent_on_mdp


def save_and_plot_return_per_episode(
        path,
        agents,
        avg_return_per_episode_per_agent,
        is_tracked_value_discounted,
        open_plot=True
):
    # Set names
    labels = [
        'episode_number', 'average_discounted_return', 'average_discounted_return_lo', 'average_discounted_return_up'
    ] if is_tracked_value_discounted else [
        'episode_number', 'average_return', 'average_return_lo', 'average_return_up'
    ]
    file_name = 'average_discounted_return_per_episode' if is_tracked_value_discounted else 'average_return_per_episode'
    x_label = r'Episode Number'
    y_label = r'Average Discounted Return' if is_tracked_value_discounted else r'Average Return'
    title_prefix = r'Average Discounted Return: ' if is_tracked_value_discounted else r'Average Return: '

    # Save
    n_episodes = len(avg_return_per_episode_per_agent[0])
    x = range(n_episodes)
    data = []
    for agent in range(len(agents)):
        data.append([])
        for episode in range(n_episodes):
            data[-1].append([
                x[episode],
                avg_return_per_episode_per_agent[agent][episode][0],
                avg_return_per_episode_per_agent[agent][episode][1],
                avg_return_per_episode_per_agent[agent][episode][2]
            ])
    save(path, csv_name=file_name, agents=agents, data=data, labels=labels)

    # Plot
    returns = []
    returns_lo = []
    returns_up = []
    for i in range(len(agents)):
        returns.append([avg_return_per_episode_per_agent[i][j][0] for j in range(n_episodes)])
        returns_lo.append([avg_return_per_episode_per_agent[i][j][1] for j in range(n_episodes)])
        returns_up.append([avg_return_per_episode_per_agent[i][j][2] for j in range(n_episodes)])
    plot(
        path, pdf_name=file_name, agents=agents, x=x, y=returns, y_lo=returns_lo, y_up=returns_up,
        x_label=x_label, y_label=y_label, title_prefix=title_prefix, open_plot=open_plot
    )


def save_and_plot_return_per_task(
        path,
        agents,
        avg_return_per_task_per_agent,
        is_tracked_value_discounted,
        open_plot=True
):
    # Set names
    labels = [
        'task_number', 'average_discounted_return', 'average_discounted_return_lo', 'average_discounted_return_up'
    ] if is_tracked_value_discounted else [
        'task_number', 'average_return', 'average_return_lo', 'average_return_up'
    ]
    file_name = 'average_discounted_return_per_task' if is_tracked_value_discounted else 'average_return_per_task'
    x_label = r'Task Number'
    y_label = r'Average Discounted Return' if is_tracked_value_discounted else r'Average Return'
    title_prefix = r'Average Discounted Return: ' if is_tracked_value_discounted else r'Average Return: '

    # Save
    n_tasks = len(avg_return_per_task_per_agent[0])
    x = range(n_tasks)
    data = []
    for agent in range(len(agents)):
        data.append([])
        for task in range(n_tasks):
            data[-1].append([
                x[task],
                avg_return_per_task_per_agent[agent][task][0],
                avg_return_per_task_per_agent[agent][task][1],
                avg_return_per_task_per_agent[agent][task][2]
            ])
    save(path, csv_name=file_name, agents=agents, data=data, labels=labels)

    # Plot
    returns = []
    returns_lo = []
    returns_up = []
    for i in range(len(agents)):
        returns.append([avg_return_per_task_per_agent[i][j][0] for j in range(n_tasks)])
        returns_lo.append([avg_return_per_task_per_agent[i][j][1] for j in range(n_tasks)])
        returns_up.append([avg_return_per_task_per_agent[i][j][2] for j in range(n_tasks)])
    plot(
        path, pdf_name=file_name, agents=agents, x=x, y=returns, y_lo=returns_lo, y_up=returns_up,
        x_label=x_label, y_label=y_label, title_prefix=title_prefix, open_plot=open_plot
    )


def run_agents_lifelong(
        agents,
        mdp_distribution,
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
        is_tracked_value_discounted=False,
        dir_for_plot='results'
):
    """
    Tweaked version of simple_rl.run_experiments.run_agents_lifelong
    Modifications are the following:
    - Tasks are first sampled so that agents experience the same sequence of tasks;
    - Track and plot return for each task with confidence interval.

    Runs each agent on the MDP distribution according to the given parameters.
    If @mdp_distribution has a non-zero horizon, then gamma is set to 1 and @steps is ignored.

    :param agents: (list)
    :param mdp_distribution: (MDPDistribution)
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
    :param is_tracked_value_discounted: (bool)
    :param dir_for_plot: (str)
    :return:
    """
    if resample_at_terminal:
        print('Warning: not implemented in this tweaked version of run_agents_lifelong')

    # Experiment (for reproducibility, plotting)
    exp_params = {"samples": samples, "episodes": episodes, "steps":steps}
    experiment = Experiment(
        agents=agents,
        mdp=mdp_distribution,
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
    avg_return_per_task_per_agent = []
    avg_return_per_episode_per_agent = []

    # Sample tasks at first so that agents experience the same sequence of tasks
    tasks = []
    for _ in range(samples):
        tasks.append(mdp_distribution.sample())

    for agent in agents:
        print(str(agent) + " is learning.")
        start = time.clock()

        # return_per_task = [(0., 0., 0.)] * samples  # Mean, lower confidence interval bound, upper
        returns = []

        for i in range(samples):
            print("  Experience task " + str(i + 1) + " of " + str(samples) + ".")

            # Select the MDP
            mdp = tasks[i]

            # Run the agent
            hit_terminal, total_steps_taken, return_per_episode = run_single_agent_on_mdp(
                agent, mdp, episodes, steps, experiment, verbose=verbose, track_disc_reward=track_disc_reward,
                reset_at_terminal=reset_at_terminal, resample_at_terminal=resample_at_terminal,
                is_tracked_value_discounted=is_tracked_value_discounted
            )

            returns.append(return_per_episode)
            # return_per_task[i] = mean_confidence_interval(return_per_episode)

            # If we re-sample at terminal, keep grabbing MDPs until we're done
            while resample_at_terminal and hit_terminal and total_steps_taken < steps:
                mdp = mdp_distribution.sample()
                hit_terminal, steps_taken, _ = run_single_agent_on_mdp(
                    agent, mdp, episodes, steps - total_steps_taken, experiment, verbose,
                    track_disc_reward, reset_at_terminal, resample_at_terminal,
                    is_tracked_value_discounted=is_tracked_value_discounted
                )
                total_steps_taken += steps_taken

            # Reset the agent
            agent.reset()

        # Store results
        avg_return_per_task = [(0., 0., 0.)] * samples  # Mean, lower, upper
        avg_return_per_episode = [(0., 0., 0.)] * episodes  # Mean, lower, upper
        for i in range(samples):
            avg_return_per_task[i] = mean_confidence_interval(returns[i])
        for j in range(episodes):
            return_per_task = [returns[i][j] for i in range(samples)]
            avg_return_per_episode[j] = mean_confidence_interval(return_per_task)

        avg_return_per_task_per_agent.append(avg_return_per_task)
        avg_return_per_episode_per_agent.append(avg_return_per_episode)

        # Track how much time this agent took
        end = time.clock()
        times[agent] = round(end - start, 3)

    # Time stuff
    print("\n--- TIMES ---")
    for agent in times.keys():
        print(str(agent) + " agent took " + str(round(times[agent], 2)) + " seconds.")
    print("-------------\n")

    # Plot
    save_and_plot_return_per_task(
        experiment.exp_directory, agents, avg_return_per_task_per_agent,
        is_tracked_value_discounted=is_tracked_value_discounted, open_plot=open_plot
    )
    save_and_plot_return_per_episode(
        experiment.exp_directory, agents, avg_return_per_episode_per_agent,
        is_tracked_value_discounted=is_tracked_value_discounted, open_plot=open_plot
    )
    experiment.make_plots(open_plot=open_plot)
