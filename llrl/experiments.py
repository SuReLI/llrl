"""
Useful functions for experiments (e.g. Lifelong RL)
"""

import time
from collections import defaultdict
from multiprocessing import Pool
from pathos.multiprocessing import ProcessPool

from llrl.utils.save import lifelong_save
from llrl.utils.chart_utils import lifelong_plot
from simple_rl.experiments import Experiment


def run_agents_lifelong(
        agents,
        mdp_distribution,
        name_identifier=None,
        n_instances=1,
        n_tasks=5,
        n_episodes=1,
        n_steps=100,
        parallel_run=False,
        clear_old_results=True,
        track_disc_reward=False,
        reset_at_terminal=False,
        cumulative_plot=True,
        dir_for_plot='results',
        verbose=False,
        plot_only=True,
        confidence=.9,
        open_plot=False,
        plot_title=True
):
    """
    Runs each agent on the MDP distribution according to the given parameters.
    If @mdp_distribution has a non-zero horizon, then gamma is set to 1 and @steps is ignored.

    :param agents: (list)
    :param mdp_distribution: (MDPDistribution)
    :param name_identifier: (str)
    :param n_instances: (int)
    :param n_tasks: (int)
    :param n_episodes: (int)
    :param n_steps: (int)
    :param parallel_run: (bool)
    :param clear_old_results: (bool)
    :param track_disc_reward: (bool) If true records and plots discounted reward, discounted over episodes.
    So, if each episode is 100 steps, then episode 2 will start discounting as though it's step 101.
    :param reset_at_terminal: (bool)
    :param cumulative_plot: (bool)
    :param dir_for_plot: (str)
    :param verbose: (bool)
    :param plot_only: (bool)
    :param confidence: (float)
    :param open_plot: (bool)
    :param plot_title: (bool)
    :return:
    """
    exp_params = {"samples": n_tasks, "episodes": n_episodes, "steps": n_steps}
    experiment = Experiment(agents=agents, mdp=mdp_distribution, name_identifier=name_identifier, params=exp_params,
                            is_episodic=n_episodes > 1, is_lifelong=True, clear_old_results=clear_old_results,
                            track_disc_reward=track_disc_reward, cumulative_plot=cumulative_plot,
                            dir_for_plot=dir_for_plot)
    path = experiment.exp_directory

    print("Running experiment:\n" + str(experiment))

    # Sample tasks
    tasks = []
    for _ in range(n_tasks):
        tasks.append(mdp_distribution.sample())

    # Run
    if not plot_only:
        '''
        if parallel_run:
            n_agents = len(agents)
            pool = ProcessPool(nodes=n_agents)
            # for i in range(n_agents):
            pool.map(
                run_single_agent_lifelong,
                agents[0], experiment, n_instances, n_tasks, n_episodes, n_steps, tasks, track_disc_reward, reset_at_terminal, verbose
            )
            pool.close()
            pool.join()
            pool.clear()
        else:
        '''
        for agent in agents:
            run_single_agent_lifelong(agent, experiment, n_instances, n_tasks, n_episodes, n_steps, tasks,
                                      track_disc_reward, reset_at_terminal, path, verbose)

    lifelong_plot(agents, path, n_tasks, n_episodes, confidence, open_plot, plot_title)


def run_single_agent_lifelong(agent, experiment, n_instances, n_tasks, n_episodes, n_steps, tasks, track_disc_reward,
                              reset_at_terminal, path, verbose):
    """
    :param agent:
    :param experiment:
    :param n_instances:
    :param n_tasks:
    :param n_episodes:
    :param n_steps:
    :param tasks:
    :param track_disc_reward:
    :param reset_at_terminal:
    :param path:
    :param verbose:
    :return:
    """
    print(str(agent) + " is learning.")
    for instance in range(1, n_instances + 1):
        agent.re_init()  # re-initialize before each instance
        data = {'returns_per_tasks': [], 'discounted_returns_per_tasks': []}

        print("  Instance " + str(instance) + " / " + str(n_instances))
        start = time.clock()
        for i in range(1, n_tasks + 1):
            print("    Experience task " + str(i) + " / " + str(n_tasks))
            task = tasks[i - 1]  # task selection

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
        lifelong_save(path, agent, data, instance, True if instance == 1 else False)


def run_single_agent_on_mdp(agent, mdp, episodes, steps, experiment=None, track_disc_reward=False,
                            reset_at_terminal=False, resample_at_terminal=False, verbose=False):
    """
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
            print("      Episode", str(episode), "/", str(episodes))

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
