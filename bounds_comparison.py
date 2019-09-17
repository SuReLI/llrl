"""
Comparison of the bounds provided by R-MAX and Lipschitz-R-Max.

Setting:
- Using two grid-world MDPs with same transition functions and different reward functions while reaching goal;
- Learning on the first MDP and transferring the Lipschitz bound to the second one;
- Plotting percentage of bound use vs amount of prior knowledge + speed-up.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
from matplotlib import rc
from tqdm import trange

from llrl.utils.env_handler import *
from llrl.utils.utils import mean_confidence_interval
from llrl.experiments import apply_async
from llrl.envs.gridworld import GridWorld
from llrl.agents.experimental.lrmax_bounds_use import ExpLRMax
from llrl.agents.experimental.rmax_bounds_use import ExpRMax
from llrl.experiments import run_agents_on_mdp


PARAM = [
    {'env': 'tight', 'n_episodes': 100, 'n_steps': 10, 'w': 11, 'h': 11, 'sto': True, 'version': 1, 'n_known': 1, 'epsilon_m': 0.01},
    {'env': 'tight', 'n_episodes': 100, 'n_steps': 10, 'w': 11, 'h': 11, 'sto': True, 'version': 1, 'n_known': 10, 'epsilon_m': 0.01},
    {'env': 'tight', 'n_episodes': 100, 'n_steps': 10, 'w': 11, 'h': 11, 'sto': False, 'version': 1, 'n_known': 1, 'epsilon_m': 0.01},
    {'env': 'tight', 'n_episodes': 100, 'n_steps': 10, 'w': 11, 'h': 11, 'sto': False, 'version': 1, 'n_known': 10, 'epsilon_m': 0.01},
    {'env': 'tight', 'n_episodes': 100, 'n_steps': 10, 'w': 11, 'h': 11, 'sto': True, 'version': 1, 'n_known': 1, 'epsilon_m': 0.00001},
    {'env': 'tight', 'n_episodes': 100, 'n_steps': 10, 'w': 11, 'h': 11, 'sto': True, 'version': 1, 'n_known': 10, 'epsilon_m': 0.00001},
    {'env': 'tight', 'n_episodes': 100, 'n_steps': 10, 'w': 11, 'h': 11, 'sto': False, 'version': 1, 'n_known': 1, 'epsilon_m': 0.00001},
    {'env': 'tight', 'n_episodes': 100, 'n_steps': 10, 'w': 11, 'h': 11, 'sto': False, 'version': 1, 'n_known': 10, 'epsilon_m': 0.00001},
    {'env': 'tight', 'n_episodes': 100, 'n_steps': 10, 'w': 11, 'h': 11, 'sto': True, 'version': 2, 'n_known': 1, 'epsilon_m': 0.01},
    {'env': 'tight', 'n_episodes': 100, 'n_steps': 10, 'w': 11, 'h': 11, 'sto': True, 'version': 2, 'n_known': 10, 'epsilon_m': 0.01},
    {'env': 'tight', 'n_episodes': 100, 'n_steps': 10, 'w': 11, 'h': 11, 'sto': False, 'version': 2, 'n_known': 1, 'epsilon_m': 0.01},
    {'env': 'tight', 'n_episodes': 100, 'n_steps': 10, 'w': 11, 'h': 11, 'sto': False, 'version': 2, 'n_known': 10, 'epsilon_m': 0.01},
    {'env': 'tight', 'n_episodes': 100, 'n_steps': 10, 'w': 11, 'h': 11, 'sto': True, 'version': 2, 'n_known': 1, 'epsilon_m': 0.00001},
    {'env': 'tight', 'n_episodes': 100, 'n_steps': 10, 'w': 11, 'h': 11, 'sto': True, 'version': 2, 'n_known': 10, 'epsilon_m': 0.00001},
    {'env': 'tight', 'n_episodes': 100, 'n_steps': 10, 'w': 11, 'h': 11, 'sto': False, 'version': 2, 'n_known': 1, 'epsilon_m': 0.00001},
    {'env': 'tight', 'n_episodes': 100, 'n_steps': 10, 'w': 11, 'h': 11, 'sto': False, 'version': 2, 'n_known': 10, 'epsilon_m': 0.00001},
    
    {'env': 'tight', 'n_episodes': 100, 'n_steps': 4, 'w': 5, 'h': 5, 'sto': True, 'version': 1, 'n_known': 1, 'epsilon_m': 0.01},
    {'env': 'tight', 'n_episodes': 100, 'n_steps': 4, 'w': 5, 'h': 5, 'sto': True, 'version': 1, 'n_known': 10, 'epsilon_m': 0.01},
    {'env': 'tight', 'n_episodes': 100, 'n_steps': 4, 'w': 5, 'h': 5, 'sto': False, 'version': 1, 'n_known': 1, 'epsilon_m': 0.01},
    {'env': 'tight', 'n_episodes': 100, 'n_steps': 4, 'w': 5, 'h': 5, 'sto': False, 'version': 1, 'n_known': 10, 'epsilon_m': 0.01},
    {'env': 'tight', 'n_episodes': 100, 'n_steps': 4, 'w': 5, 'h': 5, 'sto': True, 'version': 1, 'n_known': 1, 'epsilon_m': 0.00001},
    {'env': 'tight', 'n_episodes': 100, 'n_steps': 4, 'w': 5, 'h': 5, 'sto': True, 'version': 1, 'n_known': 10, 'epsilon_m': 0.00001},
    {'env': 'tight', 'n_episodes': 100, 'n_steps': 4, 'w': 5, 'h': 5, 'sto': False, 'version': 1, 'n_known': 1, 'epsilon_m': 0.00001},
    {'env': 'tight', 'n_episodes': 100, 'n_steps': 4, 'w': 5, 'h': 5, 'sto': False, 'version': 1, 'n_known': 10, 'epsilon_m': 0.00001},
    {'env': 'tight', 'n_episodes': 100, 'n_steps': 4, 'w': 5, 'h': 5, 'sto': True, 'version': 2, 'n_known': 1, 'epsilon_m': 0.01},
    {'env': 'tight', 'n_episodes': 100, 'n_steps': 4, 'w': 5, 'h': 5, 'sto': True, 'version': 2, 'n_known': 10, 'epsilon_m': 0.01},
    {'env': 'tight', 'n_episodes': 100, 'n_steps': 4, 'w': 5, 'h': 5, 'sto': False, 'version': 2, 'n_known': 1, 'epsilon_m': 0.01},
    {'env': 'tight', 'n_episodes': 100, 'n_steps': 4, 'w': 5, 'h': 5, 'sto': False, 'version': 2, 'n_known': 10, 'epsilon_m': 0.01},
    {'env': 'tight', 'n_episodes': 100, 'n_steps': 4, 'w': 5, 'h': 5, 'sto': True, 'version': 2, 'n_known': 1, 'epsilon_m': 0.00001},
    {'env': 'tight', 'n_episodes': 100, 'n_steps': 4, 'w': 5, 'h': 5, 'sto': True, 'version': 2, 'n_known': 10, 'epsilon_m': 0.00001},
    {'env': 'tight', 'n_episodes': 100, 'n_steps': 4, 'w': 5, 'h': 5, 'sto': False, 'version': 2, 'n_known': 1, 'epsilon_m': 0.00001},
    {'env': 'tight', 'n_episodes': 100, 'n_steps': 4, 'w': 5, 'h': 5, 'sto': False, 'version': 2, 'n_known': 10, 'epsilon_m': 0.00001},

    {'env': 'corridor', 'n_episodes': 100, 'n_steps': 11, 'w': 20, 'h': 1, 'sto': True, 'version': 0, 'n_known': 1, 'epsilon_m': 0.01},
    {'env': 'corridor', 'n_episodes': 100, 'n_steps': 11, 'w': 20, 'h': 1, 'sto': True, 'version': 0, 'n_known': 10, 'epsilon_m': 0.01},
    {'env': 'corridor', 'n_episodes': 100, 'n_steps': 11, 'w': 20, 'h': 1, 'sto': False, 'version': 0, 'n_known': 1, 'epsilon_m': 0.01},
    {'env': 'corridor', 'n_episodes': 100, 'n_steps': 11, 'w': 20, 'h': 1, 'sto': False, 'version': 0, 'n_known': 10, 'epsilon_m': 0.01},
    {'env': 'corridor', 'n_episodes': 100, 'n_steps': 11, 'w': 20, 'h': 1, 'sto': True, 'version': 0, 'n_known': 1, 'epsilon_m': 0.00001},
    {'env': 'corridor', 'n_episodes': 100, 'n_steps': 11, 'w': 20, 'h': 1, 'sto': True, 'version': 0, 'n_known': 10, 'epsilon_m': 0.00001},
    {'env': 'corridor', 'n_episodes': 100, 'n_steps': 11, 'w': 20, 'h': 1, 'sto': False, 'version': 0, 'n_known': 1, 'epsilon_m': 0.00001},
    {'env': 'corridor', 'n_episodes': 100, 'n_steps': 11, 'w': 20, 'h': 1, 'sto': False, 'version': 0, 'n_known': 10, 'epsilon_m': 0.00001}
]


def plot_bound_use(path, lrmax_path, rmax_path, n_run, confidence=.9, open_plot=False):
    lrmax_df = pd.read_csv(lrmax_path)
    rmax_df = pd.read_csv(rmax_path)

    x = []
    rlbu_m, rlbu_lo, rlbu_up = [], [], []
    su_m, su_lo, su_up = [], [], []

    for i in range(n_run):
        ldf = lrmax_df.loc[lrmax_df['run_number'] == i]
        rdf = lrmax_df.loc[rmax_df['run_number'] == i]

        prior = ldf.iloc[0]['prior']

        _rlbu_m, _rlbu_lo, _rlbu_up = mean_confidence_interval(np.array(ldf.ratio_lip_bound_use), confidence=confidence)

        lrmax_ntscv = np.array(ldf.n_time_steps_cv)
        rmax_ntscv = np.array(rdf.n_time_steps_cv)
        speed_up = 100. * (rmax_ntscv - lrmax_ntscv) / rmax_ntscv
        _su_m, _su_lo, _su_up = mean_confidence_interval(speed_up, confidence=confidence)

        x.append(prior)
        rlbu_m.append(_rlbu_m)
        rlbu_lo.append(_rlbu_lo)
        rlbu_up.append(_rlbu_up)
        su_m.append(_su_m)
        su_lo.append(_su_lo)
        su_up.append(_su_up)

        my_plot_bound_use(
            path=path,
            pdf_name='bounds_comparison',
            prior=x,
            ratio=rlbu_m,
            ratio_up=rlbu_up,
            ratio_lo=rlbu_lo,
            speed_up=su_m,
            speed_up_up=su_up,
            speed_up_lo=su_lo,
            open_plot=open_plot
        )


def my_plot_bound_use(
        path,
        pdf_name,
        prior,
        ratio,
        ratio_up,
        ratio_lo,
        speed_up,
        speed_up_up,
        speed_up_lo,
        open_plot=False,
        latex_rendering=False
):
    # LaTeX rendering
    if latex_rendering:
        rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

    colors = ['steelblue', 'darkorange', 'green']
    plt.plot(prior, ratio, '-o', color=colors[0], label=r'\% use Lipschitz bound')
    plt.fill_between(prior, ratio_up, ratio_lo, color=colors[0], alpha=0.2)
    plt.plot(prior, speed_up, '-^', color=colors[1], label=r'\% time-steps gained (speed-up)')
    plt.fill_between(prior, speed_up_up, speed_up_lo, color=colors[1], alpha=0.2)

    plt.xlim(max(prior), min(prior))  # decreasing upper-bound
    if latex_rendering:
        plt.xlabel(r'Prior knowledge (known upper-bound on $\max_{s, a} = D^{M \bar{M}}_{\gamma V^*_{\bar{M}}}(s, a)$)')
    else:
        plt.xlabel(r'Prior knowledge')
    plt.ylabel(r'\%')
    plt.legend(loc='best')
    # plt.title('')
    plt.grid(True, linestyle='--')

    # Save
    plot_file_name = os.path.join(path, pdf_name + '.pdf')
    plt.savefig(plot_file_name, format='pdf')

    # Open
    if open_plot:
        open_prefix = 'gnome-' if sys.platform == 'linux' or sys.platform == 'linux2' else ''
        os.system(open_prefix + 'open ' + plot_file_name)

    # Clear and close
    plt.cla()
    plt.close()


def sample_environments(env_class, gamma, w, h, sto=True, version=1):
    if env_class == 'grid-world':
        gl = [(w, h)]
        slip = 0.1 if sto else 0
        mdp1 = GridWorld(width=w, height=h, init_loc=(1, 1), goal_locs=gl, goal_rewards=[0.8], slip_prob=slip)
        mdp2 = GridWorld(width=w, height=h, init_loc=(1, 1), goal_locs=gl, goal_rewards=[1.0], slip_prob=slip)
    elif env_class == 'corridor':
        mdp1 = sample_corridor(gamma, 'corridor1', w=w, verbose=False, stochastic=sto)
        mdp2 = sample_corridor(gamma, 'corridor2', w=w, verbose=False, stochastic=sto)
    elif env_class == 'tight':
        mdp1 = sample_tight(gamma, 'tight1', version=version, w=w, h=h, stochastic=sto, verbose=False)
        mdp2 = sample_tight(gamma, 'tight2', version=version, w=w, h=h, stochastic=sto, verbose=False)
    else:
        raise ValueError('Error: unrecognized environment class.')
    return mdp1, mdp2


def run_twice(instance_number, run_number, rmax, lrmax, prior, mdp1, mdp2, n_episodes, n_steps):
    lrmax.prior = prior
    lrmax.re_init()
    rmax.re_init()

    lrmax.instance_number = instance_number
    lrmax.run_number = run_number
    rmax.instance_number = instance_number
    rmax.run_number = run_number

    # Run twice
    lrmax.write_data = False
    run_agents_on_mdp([lrmax], mdp1, n_instances=1, n_episodes=n_episodes, n_steps=n_steps,
                      reset_at_terminal=False, verbose=False)

    lrmax.write_data = True
    run_agents_on_mdp([lrmax, rmax], mdp2, n_instances=1, n_episodes=n_episodes, n_steps=n_steps,
                      reset_at_terminal=False, verbose=False)


def bounds_comparison_experiment(index, do_run=False, do_plot=True, multi_thread=True, n_threads=None, open_plot=False):
    p = PARAM[index]

    # Parameters
    gamma = 0.9
    n_instances = 10
    n_episodes = p['n_episodes']
    n_steps = p['n_steps']
    prior_min = 1.  # (1. + gamma) / (1. - gamma)
    prior_max = 0.
    priors = [round(p, 1) for p in np.linspace(start=prior_min, stop=prior_max, num=5)]

    # Environments
    w, h = p['w'], p['h']
    n_states = w * h
    mdp1, mdp2 = sample_environments(p['env'], gamma, w=w, h=h, sto=p['sto'], version=p['version'])

    actions = mdp1.get_actions()
    r_max = 1.
    v_max = None
    deduce_v_max = True  # erase previous definition of v_max
    n_known = p['n_known']
    epsilon_q = .01
    epsilon_m = p['epsilon_m']
    delta = .1

    # Saving parameters
    path = 'results/bounds_comparison/exp-' + str(index) + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    lrmax_path = path + 'lrmax-results.csv'
    rmax_path = path + 'rmax-results.csv'

    if do_run:
        lrmax = ExpLRMax(actions=actions, gamma=gamma, r_max=r_max, v_max=v_max, deduce_v_max=deduce_v_max,
                         n_known=n_known, deduce_n_known=False, epsilon_q=epsilon_q, epsilon_m=epsilon_m,
                         delta=delta, n_states=n_states, max_memory_size=None, prior=0.,
                         estimate_distances_online=True, min_sampling_probability=.5, name="ExpLRMax", path=lrmax_path)
        rmax = ExpRMax(actions=actions, gamma=gamma, r_max=r_max, v_max=v_max, deduce_v_max=deduce_v_max,
                       n_known=n_known, deduce_n_known=False, epsilon_q=epsilon_q, epsilon_m=epsilon_m, delta=delta,
                       n_states=n_states, name="ExpRMax", path=rmax_path)

        lrmax.write(init=True)
        rmax.write(init=True)

        if multi_thread:
            n_processes = multiprocessing.cpu_count() if n_threads is None else n_threads
            print('Using', n_processes, 'threads.')
            pool = multiprocessing.Pool(processes=n_processes)

            # Asynchronous execution
            jobs = []
            for i in range(n_instances):
                for j in range(len(priors)):
                    job = apply_async(
                        pool, run_twice, (i, j, rmax, lrmax, priors[j], mdp1, mdp2, n_episodes, n_steps)
                    )
                    jobs.append(job)

            for job in jobs:
                job.get()
        else:
            for i in trange(n_instances, desc='{:>10}'.format('instances')):
                for j in trange(len(priors), desc='{:>10}'.format('priors')):
                    run_twice(i, j, rmax, lrmax, priors[j], mdp1, mdp2, n_episodes, n_steps)
    if do_plot:
        plot_bound_use(path=path, lrmax_path=lrmax_path, rmax_path=rmax_path, n_run=len(priors), open_plot=open_plot)


if __name__ == '__main__':
    index = int(sys.argv[1])

    bounds_comparison_experiment(index, do_run=True, do_plot=True, open_plot=False)
