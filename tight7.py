"""
Lifelong RL experiment
"""

from llrl.agents.rmax import RMax
from llrl.agents.lrmax import LRMax
from llrl.agents.maxqinit import MaxQInit
from llrl.agents.lrmaxqinit import LRMaxQInit
from llrl.utils.env_handler import make_env_distribution
from llrl.experiments import run_agents_lifelong


PARAM = [
    {'name': 'tight-a', 'size': 6, 'n_tasks': 100, 'n_episodes': 200, 'n_steps': 4, 'n_known': 1, 'stochastic': False, 'v_max': 10.},
    {'name': 'tight-b', 'size': 6, 'n_tasks': 100, 'n_episodes': 200, 'n_steps': 4, 'n_known': 1, 'stochastic': False, 'v_max': 1.},
    {'name': 'tight-c', 'size': 6, 'n_tasks': 100, 'n_episodes': 200, 'n_steps': 4, 'n_known': 1, 'stochastic': True, 'v_max': 10.},
    {'name': 'tight-d', 'size': 6, 'n_tasks': 100, 'n_episodes': 200, 'n_steps': 4, 'n_known': 1, 'stochastic': True, 'v_max': 1.},
    {'name': 'tight-e', 'size': 6, 'n_tasks': 100, 'n_episodes': 200, 'n_steps': 4, 'n_known': 10, 'stochastic': True, 'v_max': 10.},
    {'name': 'tight-f', 'size': 6, 'n_tasks': 100, 'n_episodes': 200, 'n_steps': 4, 'n_known': 10, 'stochastic': True, 'v_max': 1.},

    {'name': 'tight-g', 'size': 14, 'n_tasks': 100, 'n_episodes': 200, 'n_steps': 12, 'n_known': 1, 'stochastic': False, 'v_max': 10.},
    {'name': 'tight-h', 'size': 14, 'n_tasks': 100, 'n_episodes': 200, 'n_steps': 12, 'n_known': 1, 'stochastic': False, 'v_max': 1.},
    {'name': 'tight-i', 'size': 14, 'n_tasks': 100, 'n_episodes': 200, 'n_steps': 12, 'n_known': 1, 'stochastic': True, 'v_max': 10.},
    {'name': 'tight-j', 'size': 14, 'n_tasks': 100, 'n_episodes': 200, 'n_steps': 12, 'n_known': 1, 'stochastic': True, 'v_max': 1.},
    {'name': 'tight-k', 'size': 14, 'n_tasks': 100, 'n_episodes': 200, 'n_steps': 12, 'n_known': 10, 'stochastic': True, 'v_max': 10.},
    {'name': 'tight-l', 'size': 14, 'n_tasks': 100, 'n_episodes': 200, 'n_steps': 12, 'n_known': 10, 'stochastic': True, 'v_max': 1.},

    {'name': 'tight-m', 'size': 6, 'n_tasks': 100, 'n_episodes': 100, 'n_steps': 5, 'n_known': 1, 'stochastic': False, 'v_max': 10.},
    {'name': 'tight-n', 'size': 6, 'n_tasks': 100, 'n_episodes': 100, 'n_steps': 5, 'n_known': 1, 'stochastic': False, 'v_max': 1.},
    {'name': 'tight-o', 'size': 6, 'n_tasks': 100, 'n_episodes': 100, 'n_steps': 5, 'n_known': 1, 'stochastic': True, 'v_max': 10.},
    {'name': 'tight-p', 'size': 6, 'n_tasks': 100, 'n_episodes': 100, 'n_steps': 5, 'n_known': 1, 'stochastic': True, 'v_max': 1.},
    {'name': 'tight-q', 'size': 6, 'n_tasks': 100, 'n_episodes': 100, 'n_steps': 5, 'n_known': 10, 'stochastic': True, 'v_max': 10.},
    {'name': 'tight-r', 'size': 6, 'n_tasks': 100, 'n_episodes': 100, 'n_steps': 5, 'n_known': 10, 'stochastic': True, 'v_max': 1.}
]


def experiment(p):
    # Parameters
    gamma = .9
    n_env = 5
    size = p['size']
    env_distribution = make_env_distribution(
        env_class='tight', n_env=n_env, gamma=gamma,
        env_name=p['name'],
        w=size,
        h=size,
        stochastic=p['stochastic']
    )
    actions = env_distribution.get_actions()
    n_known = p['n_known']
    p_min = 1. / float(n_env)
    epsilon_q = .01
    epsilon_m = .01
    delta = .1
    r_max = 1.
    v_max = p['v_max']
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
    run_agents_lifelong(agents_pool, env_distribution, n_instances=3, n_tasks=p['n_tasks'], n_episodes=p['n_episodes'],
                        n_steps=p['n_steps'],
                        reset_at_terminal=False, open_plot=False, plot_title=True, do_run=True, do_plot=True,
                        parallel_run=True, n_processes=None)


if __name__ == '__main__':
    # np.random.seed(1993)
    experiment(PARAM[7])
