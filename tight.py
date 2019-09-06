"""
Lifelong RL experiment in constant transition function setting
"""

import numpy as np

from llrl.agents.rmax import RMax
from llrl.agents.lrmax import LRMax
from llrl.agents.maxqinit import MaxQInit
from llrl.agents.lrmaxqinit import LRMaxQInit
from llrl.utils.env_handler import make_env_distribution
from llrl.experiments import run_agents_lifelong


def experiment():
    # Parameters
    gamma = .9
    n_env = 5
    env_distribution = make_env_distribution(env_class='tight', env_name='tight', n_env=n_env, gamma=gamma)
    actions = env_distribution.get_actions()
    n_known = 1
    p_min = 1. / float(n_env)
    epsilon_q = .01
    epsilon_m = .01
    delta = .1
    r_max = 1.
    v_max = 10.
    n_states = 4
    max_mem = 1

    # Agents
    rmax = RMax(actions=actions, gamma=gamma, r_max=r_max, v_max=v_max, deduce_v_max=False, n_known=n_known,
                deduce_n_known=False, epsilon_q=epsilon_q, epsilon_m=epsilon_m, name='RMax')
    lrmax = LRMax(actions=actions, gamma=gamma, r_max=r_max, v_max=v_max, deduce_v_max=False, n_known=n_known,
                  deduce_n_known=False, epsilon_q=epsilon_q, epsilon_m=epsilon_m, delta=delta, n_states=n_states,
                  max_memory_size=max_mem, prior=None, estimate_distances_online=True,
                  min_sampling_probability=p_min, name='LRMax')
    lrmax_p01 = LRMax(actions=actions, gamma=gamma, r_max=r_max, v_max=v_max, deduce_v_max=False, n_known=n_known,
                      deduce_n_known=False, epsilon_q=epsilon_q, epsilon_m=epsilon_m, delta=delta, n_states=n_states,
                      max_memory_size=max_mem, prior=0.1, estimate_distances_online=True,
                      min_sampling_probability=p_min, name='LRMax(Dmax=0.1)')
    lrmax_p02 = LRMax(actions=actions, gamma=gamma, r_max=r_max, v_max=v_max, deduce_v_max=False, n_known=n_known,
                      deduce_n_known=False, epsilon_q=epsilon_q, epsilon_m=epsilon_m, delta=delta, n_states=n_states,
                      max_memory_size=max_mem, prior=0.2, estimate_distances_online=True,
                      min_sampling_probability=p_min, name='LRMax(Dmax=0.2)')
    maxqinit = MaxQInit(actions=actions, gamma=gamma, r_max=r_max, v_max=v_max, deduce_v_max=False, n_known=n_known,
                        deduce_n_known=False, epsilon_q=epsilon_q, epsilon_m=epsilon_m, delta=delta, n_states=n_states,
                        min_sampling_probability=p_min, name='MaxQInit')
    lrmaxqinit = LRMaxQInit(actions=actions, gamma=gamma, r_max=r_max, v_max=v_max, deduce_v_max=False, n_known=n_known,
                            deduce_n_known=False, epsilon_q=epsilon_q, epsilon_m=epsilon_m, delta=delta,
                            n_states=n_states, max_memory_size=max_mem, prior=None, estimate_distances_online=True,
                            min_sampling_probability=p_min, name='LRMaxQInit')
    lrmaxqinit_p01 = LRMaxQInit(actions=actions, gamma=gamma, r_max=r_max, v_max=v_max, deduce_v_max=False, n_known=n_known,
                                deduce_n_known=False, epsilon_q=epsilon_q, epsilon_m=epsilon_m, delta=delta,
                                n_states=n_states, max_memory_size=max_mem, prior=0.1, estimate_distances_online=True,
                                min_sampling_probability=p_min, name='LRMaxQInit(Dmax=0.1)')
    lrmaxqinit_p02 = LRMaxQInit(actions=actions, gamma=gamma, r_max=r_max, v_max=v_max, deduce_v_max=False, n_known=n_known,
                                deduce_n_known=False, epsilon_q=epsilon_q, epsilon_m=epsilon_m, delta=delta,
                                n_states=n_states, max_memory_size=max_mem, prior=0.2, estimate_distances_online=True,
                                min_sampling_probability=p_min, name='LRMaxQInit(Dmax=0.2)')
    agents_pool = [rmax, lrmax, lrmax_p01, lrmax_p02, maxqinit, lrmaxqinit, lrmaxqinit_p01, lrmaxqinit_p02]

    # Run
    run_agents_lifelong(agents_pool, env_distribution, n_instances=3, n_tasks=100, n_episodes=200, n_steps=100,
                        reset_at_terminal=False, open_plot=False, plot_title=True, do_run=True, do_plot=True,
                        parallel_run=True, n_processes=None)


if __name__ == '__main__':
    np.random.seed(1993)
    experiment()
