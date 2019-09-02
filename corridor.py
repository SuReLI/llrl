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
    n_states = 20
    env_distribution = make_env_distribution(env_class='corridor', n_env=n_env, gamma=gamma, w=n_states, h=1)
    actions = env_distribution.get_actions()
    n_known = 1
    p_min = 1. / float(n_env)
    r_max = 1.
    v_max = 10.
    epsilon_q = .01
    epsilon_m = .01
    delta = .1
    max_mem = 1

    # Agents
    rmax = RMax(actions=actions, gamma=gamma, r_max=r_max, v_max=v_max, deduce_v_max=False, n_known=n_known,
                deduce_n_known=False, epsilon_q=epsilon_q, epsilon_m=epsilon_m, name='RMax')
    lrmax = LRMax(actions=actions, gamma=gamma, r_max=r_max, v_max=v_max, deduce_v_max=False, n_known=n_known,
                  deduce_n_known=False, epsilon_q=epsilon_q, epsilon_m=epsilon_m, delta=delta, n_states=n_states,
                  max_memory_size=max_mem, prior=None, estimate_distances_online=True,
                  min_sampling_probability=p_min, name='LRMax')
    lrmaxprior02 = LRMax(actions=actions, gamma=gamma, r_max=r_max, v_max=v_max, deduce_v_max=False, n_known=n_known,
                         deduce_n_known=False, epsilon_q=epsilon_q, epsilon_m=epsilon_m, delta=delta, n_states=n_states,
                         max_memory_size=max_mem, prior=0.2, estimate_distances_online=False,
                         min_sampling_probability=p_min, name='LRMax(Dmax0.2)')
    maxqinit = MaxQInit(actions=actions, gamma=gamma, r_max=r_max, v_max=v_max, deduce_v_max=False, n_known=n_known,
                        deduce_n_known=False, epsilon_q=epsilon_q, epsilon_m=epsilon_m, delta=delta, n_states=n_states,
                        min_sampling_probability=p_min, name='MaxQInit')
    lrmaxqinit = LRMaxQInit(actions=actions, gamma=gamma, r_max=r_max, v_max=v_max, deduce_v_max=False, n_known=n_known,
                            deduce_n_known=False, epsilon_q=epsilon_q, epsilon_m=epsilon_m, delta=delta,
                            n_states=n_states, max_memory_size=max_mem, prior=None, estimate_distances_online=True,
                            min_sampling_probability=p_min, name='LRMaxQInit')
    lrmaxqinitprior02 = LRMaxQInit(actions=actions, gamma=gamma, r_max=r_max, v_max=v_max, deduce_v_max=False, n_known=n_known,
                                   deduce_n_known=False, epsilon_q=epsilon_q, epsilon_m=epsilon_m, delta=delta,
                                   n_states=n_states, max_memory_size=max_mem, prior=0.2, estimate_distances_online=True,
                                   min_sampling_probability=p_min, name='LRMaxQInit(Dmax0.2)')
    agents_pool = [rmax, lrmax, lrmaxprior02, maxqinit, lrmaxqinit, lrmaxqinitprior02]

    # Run
    run_agents_lifelong(agents_pool, env_distribution, name_identifier=None, n_instances=1, n_tasks=20, n_episodes=20,
                        n_steps=11, reset_at_terminal=False, open_plot=True, plot_title=True,
                        do_run=True, do_plot=True,
                        parallel_run=True, n_processes=None)


if __name__ == '__main__':
    np.random.seed(1993)
    experiment()
