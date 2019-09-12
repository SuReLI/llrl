"""
Useful functions for experiments (e.g. Lifelong RL)
"""

import time
import dill
import multiprocessing
from collections import defaultdict

from llrl.utils.save import lifelong_save, save_script
from llrl.utils.chart_utils import lifelong_plot
from simple_rl.experiments import Experiment


def run_dill_encoded(payload):
    fun, args = dill.loads(payload)
    return fun(*args)


def apply_async(pool, fun, args):
    payload = dill.dumps((fun, args))
    return pool.apply_async(run_dill_encoded, (payload,))


def run_agents_lifelong(
        agents,
        mdp_distribution,
        name_identifier=None,
        n_instances=1,
        n_tasks=5,
        n_episodes=1,
        n_steps=100,
        parallel_run=False,
        n_processes=None,
        clear_old_results=True,
        track_disc_reward=False,
        reset_at_terminal=False,
        cumulative_plot=True,
        dir_for_plot='results',
        verbose=False,
        do_run=True,
        do_plot=False,
        confidence=.9,
        open_plot=False,
        plot_title=True,
        plot_legend=True,
        episodes_moving_average=False,
        episodes_ma_width=10,
        tasks_moving_average=False,
        tasks_ma_width=10,
        latex_rendering=False
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
    :param n_processes: (int)
    :param clear_old_results: (bool)
    :param track_disc_reward: (bool) If true records and plots discounted reward, discounted over episodes.
    So, if each episode is 100 steps, then episode 2 will start discounting as though it's step 101.
    :param reset_at_terminal: (bool)
    :param cumulative_plot: (bool)
    :param dir_for_plot: (str)
    :param verbose: (bool)
    :param do_run: (bool)
    :param do_plot: (bool)
    :param confidence: (float)
    :param open_plot: (bool)
    :param plot_title: (bool)
    :param plot_legend: (bool)
    :param episodes_moving_average: (bool)
    :param episodes_ma_width: (int)
    :param tasks_moving_average: (bool)
    :param tasks_ma_width: (int)
    :param latex_rendering: (bool)
    :return:
    """
    exp_params = {"samples": n_tasks, "episodes": n_episodes, "steps": n_steps}
    experiment = Experiment(agents=agents, mdp=mdp_distribution, name_identifier=name_identifier, params=exp_params,
                            is_episodic=n_episodes > 1, is_lifelong=True, clear_old_results=clear_old_results,
                            track_disc_reward=track_disc_reward, cumulative_plot=cumulative_plot,
                            dir_for_plot=dir_for_plot)
    path = experiment.exp_directory
    save_script(path)

    print("Running experiment:\n" + str(experiment))

    # Sample tasks
    tasks = []
    for _ in range(n_tasks):
        tasks.append(mdp_distribution.sample())
    n_agents = len(agents)

    # Run
    if do_run:
        if parallel_run:
            n_processes = multiprocessing.cpu_count() if n_processes is None else n_processes
            print('Using', n_processes, 'threads.')
            pool = multiprocessing.Pool(processes=n_processes)

            # Asynchronous execution
            jobs = []
            for i in range(n_agents):
                lifelong_save(init=True, path=path, agent=agents[i])
                for j in range(n_instances):
                    job = apply_async(
                        pool, run_agent_lifelong,
                        (agents[i], experiment, j, n_tasks, n_episodes, n_steps, tasks, track_disc_reward,
                         reset_at_terminal, path, verbose)
                    )
                    jobs.append(job)

            for job in jobs:
                job.get()
        else:
            for i in range(n_agents):
                lifelong_save(init=True, path=path, agent=agents[i])
                for j in range(n_instances):
                    run_agent_lifelong(agents[i], experiment, j, n_tasks, n_episodes, n_steps, tasks, track_disc_reward,
                                       reset_at_terminal, path, verbose)

    # Plot
    if do_plot:
        lifelong_plot(agents, path, n_tasks, n_episodes, confidence, open_plot, plot_title, plot_legend,
                      episodes_moving_average=episodes_moving_average, episodes_ma_width=episodes_ma_width,
                      tasks_moving_average=tasks_moving_average, tasks_ma_width=tasks_ma_width,
                      latex_rendering=latex_rendering)


def multi_instances_run_agent_lifelong(agent, experiment, parallel_run, n_parallel_instances, n_instances, n_tasks,
                                       n_episodes, n_steps, tasks, track_disc_reward, reset_at_terminal, path, verbose):
    """
    :param agent: ()
    :param experiment: ()
    :param parallel_run: (bool)
    :param n_parallel_instances: (int)
    :param n_instances: (int)
    :param n_tasks: (int)
    :param n_episodes: (int)
    :param n_steps: (int)
    :param tasks: (list)
    :param track_disc_reward: (bool)
    :param reset_at_terminal: (bool)
    :param path: (str)
    :param verbose: (bool)
    :return: None
    """
    print(str(agent) + " is learning.")

    # Initialize save
    lifelong_save(init=True, path=path, agent=agent)

    if parallel_run:
        for instance in range(1, n_parallel_instances + 1):
            print("  Instance " + str(instance) + " / " + str(n_parallel_instances))
    else:
        for instance in range(1, n_instances + 1):
            print("  Instance " + str(instance) + " / " + str(n_instances))
            run_agent_lifelong(agent, experiment, instance, n_tasks, n_episodes, n_steps, tasks, track_disc_reward,
                               reset_at_terminal, path, verbose)


def run_agent_lifelong(agent, experiment, instance_number, n_tasks, n_episodes, n_steps, tasks, track_disc_reward, reset_at_terminal,
                       path, verbose):
    """
    :param agent: ()
    :param experiment: ()
    :param instance_number: (int)
    :param n_tasks: (int)
    :param n_episodes: (int)
    :param n_steps: (int)
    :param tasks: (list)
    :param track_disc_reward: (bool)
    :param reset_at_terminal: (bool)
    :param path: (str)
    :param verbose: (bool)
    :return: None
    """
    agent.re_init()  # re-initialize before each instance
    data = {'returns_per_tasks': [], 'discounted_returns_per_tasks': []}

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
    lifelong_save(init=False, path=path, agent=agent, data=data, instance_number=instance_number)


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


def run_single_agent_on_mdp(agent, mdp, n_episodes, n_steps, experiment=None, track_disc_reward=False,
                            reset_at_terminal=False, resample_at_terminal=False, verbose=False):
    """
    :param agent:
    :param mdp:
    :param n_episodes:
    :param n_steps:
    :param experiment:
    :param track_disc_reward:
    :param reset_at_terminal:
    :param resample_at_terminal:
    :param verbose:
    :return:
    """
    if reset_at_terminal and resample_at_terminal:
        raise ValueError("ExperimentError: Can't have reset_at_terminal and resample_at_terminal set to True.")

    return_per_episode = [0] * n_episodes
    discounted_return_per_episode = [0] * n_episodes
    gamma = mdp.get_gamma()

    # For each episode.
    for episode in range(1, n_episodes + 1):
        cumulative_episodic_reward = 0.

        if verbose:
            print("      Episode", str(episode), "/", str(n_episodes))

        # Compute initial state/reward.
        state = mdp.get_init_state()
        reward = 0.

        for step in range(1, n_steps + 1):

            # step time
            step_start = time.clock()

            # Compute the agent's policy.
            action = agent.act(state, reward)

            # Terminal check.
            if state.is_terminal():
                if n_episodes == 1 and not reset_at_terminal and experiment is not None and action != "terminate":
                    # Self loop if we're not episodic or resetting and in a terminal state.
                    experiment.add_experience(agent, state, action, 0, state, time_taken=time.clock()-step_start)
                    continue
                break

            # Execute in MDP.
            reward, next_state = mdp.execute_agent_action(action)

            # Track value.
            return_per_episode[episode - 1] += reward
            discounted_return_per_episode[episode - 1] += reward * (gamma ** step)
            cumulative_episodic_reward += reward

            # Record the experience.
            if experiment is not None:
                reward_to_track = mdp.get_gamma()**(step + 1 + episode*n_steps) * reward if track_disc_reward else reward
                reward_to_track = round(reward_to_track, 5)
                experiment.add_experience(agent, state, action, reward_to_track, next_state,
                                          time_taken=time.clock() - step_start)

            if next_state.is_terminal():
                if reset_at_terminal:
                    # Reset the MDP.
                    next_state = mdp.get_init_state()
                    mdp.reset()
                elif resample_at_terminal and step < n_steps:
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

    return False, n_steps, return_per_episode, discounted_return_per_episode


def run_agents_on_mdp(agents, mdp, n_instances, n_episodes, n_steps, clear_old_results=True, track_disc_reward=False,
                      verbose=False, reset_at_terminal=False, cumulative_plot=True, dir_for_plot="results",
                      experiment_name_prefix="", name_identifier=None, track_success=False, success_reward=None):
    """
    Run each agent of a list of agents in a single environment.
    TODO implement save / plot routines if needed.

    :param agents:
    :param mdp:
    :param n_instances:
    :param n_episodes:
    :param n_steps:
    :param clear_old_results:
    :param track_disc_reward:
    :param open_plot:
    :param verbose:
    :param reset_at_terminal:
    :param cumulative_plot:
    :param dir_for_plot:
    :param experiment_name_prefix:
    :param name_identifier:
    :param track_success:
    :param success_reward:
    :return: None
    """
    if track_success and success_reward is None:
        raise ValueError("(simple_rl): run_agents_on_mdp must set param @success_reward when @track_success=True.")

    exp_params = {"instances":n_instances, "episodes": n_episodes, "steps": n_steps}
    experiment = Experiment(agents=agents, mdp=mdp, name_identifier=name_identifier, params=exp_params,
                            is_episodic=n_episodes > 1, clear_old_results=clear_old_results,
                            track_disc_reward=track_disc_reward, cumulative_plot=cumulative_plot,
                            dir_for_plot=dir_for_plot, experiment_name_prefix=experiment_name_prefix,
                            track_success=track_success, success_reward=success_reward)

    # Record how long each agent spends learning.
    print("Running experiment: \n" + str(experiment))
    time_dict = defaultdict(float)

    # Learn.
    for agent in agents:
        print(str(agent) + " is learning.")

        start = time.clock()

        # For each instance.
        for instance in range(1, n_instances + 1):
            print("  Instance " + str(instance) + " / " + str(n_instances))

            # Run on task
            _, _, returns, discounted_returns = run_single_agent_on_mdp(
                agent, mdp, n_episodes=n_episodes, n_steps=n_steps, experiment=experiment, verbose=verbose,
                track_disc_reward=track_disc_reward, reset_at_terminal=reset_at_terminal, resample_at_terminal=False
            )

            # Reset between each instance
            agent.reset()
            # mdp.end_of_instance()

        # Track how much time this agent took.
        end = time.clock()
        time_dict[agent] = round(end - start, 3)

    # Time stuff.
    print("Elapsed times:")
    for agent in time_dict.keys():
        print(str(agent) + " agent took " + str(round(time_dict[agent], 2)) + " seconds.")
