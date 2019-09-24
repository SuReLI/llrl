"""
Lifelong RL experiment
"""


import sys


from llrl.agents.rmax import RMax
from llrl.agents.lrmax import LRMax
from llrl.agents.maxqinit import MaxQInit
from llrl.agents.lrmaxqinit import LRMaxQInit
from llrl.utils.env_handler import make_env_distribution
from llrl.experiments import run_agents_lifelong

PARAM = [
    {'version': 2, 'size': 6, 'n_tasks': 20, 'n_episodes': 50, 'n_steps': 4, 'n_known': 1, 'stochastic': False},
    {'version': 2, 'size': 6, 'n_tasks': 20, 'n_episodes': 50, 'n_steps': 4, 'n_known': 1, 'stochastic': True},
    {'version': 2, 'size': 6, 'n_tasks': 20, 'n_episodes': 150, 'n_steps': 4, 'n_known': 3, 'stochastic': True},
    {'version': 2, 'size': 6, 'n_tasks': 20, 'n_episodes': 500, 'n_steps': 4, 'n_known': 10, 'stochastic': True},

    {'version': 2, 'size': 6, 'n_tasks': 20, 'n_episodes': 50, 'n_steps': 7, 'n_known': 1, 'stochastic': False},
    {'version': 2, 'size': 6, 'n_tasks': 20, 'n_episodes': 50, 'n_steps': 7, 'n_known': 1, 'stochastic': True},
    {'version': 2, 'size': 6, 'n_tasks': 20, 'n_episodes': 150, 'n_steps': 7, 'n_known': 3, 'stochastic': True},
    {'version': 2, 'size': 6, 'n_tasks': 20, 'n_episodes': 500, 'n_steps': 7, 'n_known': 10, 'stochastic': True},

    {'version': 2, 'size': 11, 'n_tasks': 20, 'n_episodes': 200, 'n_steps': 10, 'n_known': 1, 'stochastic': False},
    {'version': 2, 'size': 11, 'n_tasks': 30, 'n_episodes': 400, 'n_steps': 10, 'n_known': 1, 'stochastic': True},
    {'version': 2, 'size': 11, 'n_tasks': 30, 'n_episodes': 1200, 'n_steps': 10, 'n_known': 3, 'stochastic': True},
    {'version': 2, 'size': 11, 'n_tasks': 15, 'n_episodes': 2000, 'n_steps': 10, 'n_known': 10, 'stochastic': True},  # 11 -> BIS
    {'version': 2, 'size': 11, 'n_tasks': 30, 'n_episodes': 4000, 'n_steps': 10, 'n_known': 10, 'stochastic': True},  # 11 -> SELECTED

    {'version': 2, 'size': 14, 'n_tasks': 20, 'n_episodes': 200, 'n_steps': 12, 'n_known': 1, 'stochastic': False},
    {'version': 2, 'size': 14, 'n_tasks': 30, 'n_episodes': 400, 'n_steps': 12, 'n_known': 1, 'stochastic': True},
    {'version': 2, 'size': 14, 'n_tasks': 30, 'n_episodes': 1200, 'n_steps': 12, 'n_known': 3, 'stochastic': True},
    {'version': 2, 'size': 14, 'n_tasks': 30, 'n_episodes': 4000, 'n_steps': 12, 'n_known': 10, 'stochastic': True}  # FAIL
]


def experiment(p, name):
    # Parameters
    gamma = .9
    n_env = 5
    size = p['size']
    env_distribution = make_env_distribution(
        env_class='tight', n_env=n_env, gamma=gamma,
        env_name=name,
        version=p['version'],
        w=size,
        h=size,
        stochastic=p['stochastic'],
        verbose=False
    )
    actions = env_distribution.get_actions()
    n_known = p['n_known']
    p_min = 1. / n_env
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
                      min_sampling_probability=p_min, name='LRMax(0.1)')
    lrmax_p015 = LRMax(actions=actions, gamma=gamma, r_max=r_max, v_max=v_max, deduce_v_max=False, n_known=n_known,
                       deduce_n_known=False, epsilon_q=epsilon_q, epsilon_m=epsilon_m, delta=delta, n_states=n_states,
                       max_memory_size=max_mem, prior=0.15, estimate_distances_online=True,
                       min_sampling_probability=p_min, name='LRMax(0.15)')
    lrmax_p02 = LRMax(actions=actions, gamma=gamma, r_max=r_max, v_max=v_max, deduce_v_max=False, n_known=n_known,
                      deduce_n_known=False, epsilon_q=epsilon_q, epsilon_m=epsilon_m, delta=delta, n_states=n_states,
                      max_memory_size=max_mem, prior=0.2, estimate_distances_online=True,
                      min_sampling_probability=p_min, name='LRMax(0.2)')
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
                                min_sampling_probability=p_min, name='LRMaxQInit(0.1)')
    lrmaxqinit_p015 = LRMaxQInit(actions=actions, gamma=gamma, r_max=r_max, v_max=v_max, deduce_v_max=False, n_known=n_known,
                                 deduce_n_known=False, epsilon_q=epsilon_q, epsilon_m=epsilon_m, delta=delta,
                                 n_states=n_states, max_memory_size=max_mem, prior=0.15, estimate_distances_online=True,
                                 min_sampling_probability=p_min, name='LRMaxQInit(0.15)')
    lrmaxqinit_p02 = LRMaxQInit(actions=actions, gamma=gamma, r_max=r_max, v_max=v_max, deduce_v_max=False, n_known=n_known,
                                deduce_n_known=False, epsilon_q=epsilon_q, epsilon_m=epsilon_m, delta=delta,
                                n_states=n_states, max_memory_size=max_mem, prior=0.2, estimate_distances_online=True,
                                min_sampling_probability=p_min, name='LRMaxQInit(0.2)')
    # agents_pool = [rmax, lrmax, lrmax_p01, lrmax_p015, lrmax_p02, maxqinit, lrmaxqinit, lrmaxqinit_p01, lrmaxqinit_p015, lrmaxqinit_p02]
    agents_pool = [rmax, lrmax, lrmax_p02, lrmax_p01, maxqinit, lrmaxqinit, lrmaxqinit_p01]

    # Run
    run_agents_lifelong(agents_pool, env_distribution, n_instances=2, n_tasks=p['n_tasks'], n_episodes=p['n_episodes'],
                        n_steps=p['n_steps'], reset_at_terminal=False, open_plot=False, plot_title=False,
                        plot_legend=2, do_run=False, do_plot=True, parallel_run=True, n_processes=None,
                        episodes_moving_average=True, episodes_ma_width=100, tasks_moving_average=False,
                        latex_rendering=True)


if __name__ == '__main__':
    # np.random.seed(1993)

    experiment_index = int(sys.argv[1])
    tight_version = PARAM[experiment_index]['version']
    experiment_name = 'tight-v' + str(tight_version) + '-' + str(experiment_index)

    experiment(PARAM[experiment_index], experiment_name)
