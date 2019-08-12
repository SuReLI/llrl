"""
Useful functions for making experiments (e.g. Lifelong RL)
"""

import sys
import time
from collections import defaultdict

from llrl.utils.utils import mean_confidence_interval
from llrl.utils.save import lifelong_save, save_agents, open_agents
from llrl.utils.chart_utils import plot
from simple_rl.experiments import Experiment


def plot_return_per_episode(
        path,
        agents,
        is_tracked_value_discounted,
        open_plot=True,
        plot_title=True
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

    # Open data
    data_frames = open_agents(path, csv_name=file_name, agents=agents)

    # Plot
    n_episodes = len(data_frames[0][labels[0]])
    x = range(n_episodes)
    returns = []
    returns_lo = []
    returns_up = []
    for df in data_frames:
        returns.append(df[labels[1]][0:n_episodes])
        returns_lo.append(df[labels[2]][0:n_episodes])
        returns_up.append(df[labels[3]][0:n_episodes])
    plot(
        path, pdf_name=file_name, agents=agents, x=x, y=returns, y_lo=returns_lo, y_up=returns_up,
        x_label=x_label, y_label=y_label, title_prefix=title_prefix, open_plot=open_plot, plot_title=plot_title
    )


def save_return_per_task(
        path,
        agents,
        avg_return_per_task_per_agent,
        is_tracked_value_discounted
):
    # Set names
    labels = [
        'task_number', 'average_discounted_return', 'average_discounted_return_lo', 'average_discounted_return_up'
    ] if is_tracked_value_discounted else [
        'task_number', 'average_return', 'average_return_lo', 'average_return_up'
    ]
    file_name = 'average_discounted_return_per_task' if is_tracked_value_discounted else 'average_return_per_task'

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
    save_agents(path, csv_name=file_name, agents=agents, data=data, labels=labels)


def plot_return_per_task(
        path,
        agents,
        is_tracked_value_discounted,
        open_plot=True,
        plot_title=True
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

    # Open data
    data_frames = open_agents(path, csv_name=file_name, agents=agents)

    # Plot
    n_tasks = len(data_frames[0][labels[0]])
    x = range(n_tasks)
    returns = []
    returns_lo = []
    returns_up = []
    for df in data_frames:
        returns.append(df[labels[1]][0:n_tasks])
        returns_lo.append(df[labels[2]][0:n_tasks])
        returns_up.append(df[labels[3]][0:n_tasks])
    plot(
        path, pdf_name=file_name, agents=agents, x=x, y=returns, y_lo=returns_lo, y_up=returns_up,
        x_label=x_label, y_label=y_label, title_prefix=title_prefix, open_plot=open_plot, plot_title=plot_title
    )


def run_agents_lifelong(
        agents,
        mdp_distribution,
        n_instances=1,
        n_tasks=5,
        n_episodes=1,
        n_steps=100,
        clear_old_results=True,
        open_plot=True,
        track_disc_reward=False,
        reset_at_terminal=False,
        cumulative_plot=True,
        is_tracked_value_discounted=False,
        confidence=.9,
        plot_only=False,
        plot_title=True,
        dir_for_plot='results',
        verbose=False
):
    """
    Runs each agent on the MDP distribution according to the given parameters.
    If @mdp_distribution has a non-zero horizon, then gamma is set to 1 and @steps is ignored.

    :param agents: (list)
    :param mdp_distribution: (MDPDistribution)
    :param n_instances: (int)
    :param n_tasks: (int)
    :param n_episodes: (int)
    :param n_steps: (int)
    :param clear_old_results: (bool)
    :param open_plot: (bool)
    :param track_disc_reward: (bool) If true records and plots discounted reward, discounted over episodes.
    So, if each episode is 100 steps, then episode 2 will start discounting as though it's step 101.
    :param reset_at_terminal: (bool)
    :param cumulative_plot: (bool)
    :param plot_only: (bool)
    :param plot_title: (bool)
    :param is_tracked_value_discounted: (bool)
    :param confidence: (float)
    :param dir_for_plot: (str)
    :param verbose: (bool)
    :return:
    """
    exp_params = {"samples": n_tasks, "episodes": n_episodes, "steps": n_steps}
    experiment = Experiment(agents=agents, mdp=mdp_distribution, params=exp_params, is_episodic=n_episodes > 1,
                            is_lifelong=True, clear_old_results=clear_old_results, track_disc_reward=track_disc_reward,
                            cumulative_plot=cumulative_plot, dir_for_plot=dir_for_plot)

    if not plot_only:
        print("Running experiment:\n" + str(experiment))
        avg_return_per_task_per_agent = []
        avg_return_per_episode_per_agent = []

        # Sample tasks at first so that agents experience the same sequence of tasks
        tasks = []
        for _ in range(n_tasks):
            tasks.append(mdp_distribution.sample())

        for agent in agents:
            run_single_agent_lifelong(agent)

        # Save
        save_return_per_task(experiment.exp_directory, agents, avg_return_per_task_per_agent,
                             is_tracked_value_discounted=is_tracked_value_discounted)
        save_return_per_episode(experiment.exp_directory, agents, avg_return_per_episode_per_agent,
                                is_tracked_value_discounted=is_tracked_value_discounted)
        # experiment.make_plots(open_plot=open_plot)

    # Plot
    plot_return_per_task(experiment.exp_directory, agents, is_tracked_value_discounted=is_tracked_value_discounted,
                         open_plot=open_plot, plot_title=plot_title)
    plot_return_per_episode(experiment.exp_directory, agents, is_tracked_value_discounted=is_tracked_value_discounted,
                            open_plot=open_plot, plot_title=plot_title)


def run_single_agent_lifelong(agent, experiment, n_instances, n_tasks, n_episodes, n_steps, tasks, track_disc_reward,
                              reset_at_terminal, verbose):
    """"""
    print(str(agent) + " is learning.")
    for instance in range(1, n_instances + 1):
        agent.re_init()  # re-initialize before each instance
        data = {'returns_per_tasks': [], 'discounted_returns_per_tasks': []}

        print("  Instance " + str(instance) + " / " + n_instances)
        start = time.clock()
        for i in range(1, n_tasks + 1):
            print("    Experience task " + str(i) + " / " + n_tasks)
            task = tasks[i]  # task selection

            # Run on task
            _, _, returns, discounted_returns = run_single_agent_on_mdp(
                agent, task, n_episodes, n_steps, experiment, verbose=verbose, track_disc_reward=track_disc_reward,
                reset_at_terminal=reset_at_terminal, resample_at_terminal=False
            )

            # Store
            data['returns_per_tasks'].append(returns)
            data['discounted_returns_per_tasks'].append(discounted_returns)

            # Reset the agent
            agent.reset()
        print("    Total time elapsed: " + str(round(time.clock() - start, 3)))

        # Save
        lifelong_save(experiment.exp_directory, agent, data, instance, True if instance == 1 else False)

        # TODO remove
        '''
        avg_return_per_task = [(0., 0., 0.)] * n_tasks  # Mean, lower, upper
        avg_return_per_episode = [(0., 0., 0.)] * n_episodes  # Mean, lower, upper
        for i in range(n_tasks):
            avg_return_per_task[i] = mean_confidence_interval(returns[i], confidence=confidence)
        for j in range(n_episodes):
            return_per_task = [returns[i][j] for i in range(n_tasks)]
            avg_return_per_episode[j] = mean_confidence_interval(return_per_task, confidence=confidence)
            
        avg_return_per_task_per_agent.append(avg_return_per_task)
        avg_return_per_episode_per_agent.append(avg_return_per_episode)
        '''


def run_single_agent_on_mdp(agent, mdp, episodes, steps, experiment=None, track_disc_reward=False,
                            reset_at_terminal=False, resample_at_terminal=False, verbose=False):
    """
    TODO
    :param agent:
    :param mdp:
    :param episodes:
    :param steps:
    :param experiment:
    :param track_disc_reward:
    :param reset_at_terminal:
    :param resample_at_terminal:
    :param verbose:
    :return:
    """
    if reset_at_terminal and resample_at_terminal:
        raise ValueError("ExperimentError: Can't have reset_at_terminal and resample_at_terminal set to True.")

    return_per_episode = [0] * episodes
    discounted_return_per_episode = [0] * episodes
    gamma = mdp.get_gamma()

    # For each episode.
    for episode in range(1, episodes + 1):
        cumulative_episodic_reward = 0.

        if verbose:
            print("      Episode", episode, "/", episodes)

        # Compute initial state/reward.
        state = mdp.get_init_state()
        reward = 0.

        for step in range(1, steps + 1):

            # step time
            step_start = time.clock()

            # Compute the agent's policy.
            action = agent.act(state, reward)

            # Terminal check.
            if state.is_terminal():
                if episodes == 1 and not reset_at_terminal and experiment is not None and action != "terminate":
                    # Self loop if we're not episodic or resetting and in a terminal state.
                    experiment.add_experience(agent, state, action, 0, state, time_taken=time.clock()-step_start)
                    continue
                break

            # Execute in MDP.
            reward, next_state = mdp.execute_agent_action(action)

            # Track value.
            return_per_episode[episode - 1] += reward * (gamma ** step)
            discounted_return_per_episode[episode - 1] += reward
            cumulative_episodic_reward += reward

            # Record the experience.
            if experiment is not None:
                reward_to_track = mdp.get_gamma()**(step + 1 + episode*steps) * reward if track_disc_reward else reward
                reward_to_track = round(reward_to_track, 5)
                experiment.add_experience(agent, state, action, reward_to_track, next_state,
                                          time_taken=time.clock() - step_start)

            if next_state.is_terminal():
                if reset_at_terminal:
                    # Reset the MDP.
                    next_state = mdp.get_init_state()
                    mdp.reset()
                elif resample_at_terminal and step < steps:
                    mdp.reset()
                    return True, step, return_per_episode, discounted_return_per_episode

            # Update pointer.
            state = next_state

        # A final update.
        _ = agent.act(state, reward)

        # Process experiment info at end of episode.
        if experiment is not None:
            experiment.end_of_episode(agent)

        # Reset the MDP, tell the agent the episode is over.
        mdp.reset()
        agent.end_of_episode()

    # Process that learning instance's info at end of learning.
    if experiment is not None:
        experiment.end_of_instance(agent)

    return False, steps, return_per_episode, discounted_return_per_episode
