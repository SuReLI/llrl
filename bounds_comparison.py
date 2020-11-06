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
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from matplotlib import rc
from tqdm import trange

from llrl.utils.env_handler import sample_corridor, sample_tight
from llrl.utils.utils import mean_confidence_interval
from llrl.experiments import apply_async
from llrl.envs.gridworld import GridWorld
from llrl.agents.experimental.lrmax_bounds_use import ExpLRMax
from llrl.agents.experimental.rmax_bounds_use import ExpRMax
from llrl.experiments import run_agents_on_mdp


RGB_COLORS_LST = [
    [153, 194, 255],

    [159, 198, 177],
    [255, 102, 102],
    [128, 179, 151],
    [96, 160, 126],

    [90, 90, 90],

    [255, 166, 77],
    [255, 102, 102]
]


PARAM = [
    {'env': 'tight', 'n_episodes': 2000, 'n_steps': 10, 'w': 11, 'h': 11, 'sto': True, 'version': 2, 'n_known': 10, 'epsilon_m': 0.01},
    {'env': 'tight', 'n_episodes': 2000, 'n_steps': 10, 'w': 11, 'h': 11, 'sto': True, 'version': 2, 'n_known': 10, 'epsilon_m': 0.001},
    {'env': 'tight', 'n_episodes': 2000, 'n_steps': 10, 'w': 11, 'h': 11, 'sto': True, 'version': 2, 'n_known': 10, 'epsilon_m': 0.0001},
    {'env': 'tight', 'n_episodes': 2000, 'n_steps': 10, 'w': 11, 'h': 11, 'sto': True, 'version': 2, 'n_known': 10, 'epsilon_m': 0.00001},
    {'env': 'tight', 'n_episodes': 2000, 'n_steps': 10, 'w': 11, 'h': 11, 'sto': True, 'version': 2, 'n_known': 10, 'epsilon_m': 0.000001}
]


def compute_speed_up(m, lo, up, lrmax_df, rmax_df, confidence=0.9, rmax_m_lrmax=True):
    lrmax_data = np.array(lrmax_df)
    rmax_data = np.array(rmax_df)
    speed_up = np.zeros(shape=rmax_data.shape)
    for i in range(len(speed_up)):
        if rmax_data[i] < 1e-10:
            rmax_data[i] = 0.
        diff = (rmax_data[i] - lrmax_data[i]) if rmax_m_lrmax else - (rmax_data[i] - lrmax_data[i])
        speed_up[i] = 100. if rmax_data[i] == 0. else 100. * diff / rmax_data[i]
    _su_m, _su_lo, _su_up = mean_confidence_interval(speed_up, confidence=confidence)
    m.append(_su_m)
    lo.append(_su_lo)
    up.append(_su_up)
    return m, lo, up


def plot_bound_use(path, lrmax_path, rmax_path, n_run, confidence=0.9, open_plot=False):
    lrmax_df = pd.read_csv(lrmax_path)
    rmax_df = pd.read_csv(rmax_path)

    x = []

    rlbu_m, rlbu_lo, rlbu_up = [], [], []
    su_m, su_lo, su_up = [], [], []
    su_t2_m, su_t2_lo, su_t2_up = [], [], []
    su_t5_m, su_t5_lo, su_t5_up = [], [], []
    su_t10_m, su_t10_lo, su_t10_up = [], [], []
    su_t50_m, su_t50_lo, su_t50_up = [], [], []
    tr_m, tr_lo, tr_up = [], [], []
    dr_m, dr_lo, dr_up = [], [], []

    for i in range(n_run):
        ldf = lrmax_df.loc[lrmax_df['run_number'] == i]
        rdf = rmax_df.loc[rmax_df['run_number'] == i]

        # Prior
        prior = ldf.iloc[0]['prior']
        x.append(prior)

        # Ratio Lipschitz bound use
        _rlbu_m, _rlbu_lo, _rlbu_up = mean_confidence_interval(100. * np.array(ldf.ratio_lip_bound_use), confidence=confidence)
        rlbu_m.append(_rlbu_m)
        rlbu_lo.append(_rlbu_lo)
        rlbu_up.append(_rlbu_up)

        # Total speed-up
        su_m, su_lo, su_up = compute_speed_up(su_m, su_lo, su_up, ldf.n_time_steps_cv, rdf.n_time_steps_cv)

        # Average speed-up 2 ts
        su_t2_m, su_t2_lo, su_t2_up = compute_speed_up(su_t2_m, su_t2_lo, su_t2_up, ldf.avg_ts_l2, rdf.avg_ts_l2)

        # Average speed-up 5 ts
        su_t5_m, su_t5_lo, su_t5_up = compute_speed_up(su_t5_m, su_t5_lo, su_t5_up, ldf.avg_ts_l5, rdf.avg_ts_l5)

        # Average speed-up 10 ts
        su_t10_m, su_t10_lo, su_t10_up = compute_speed_up(su_t10_m, su_t10_lo, su_t10_up, ldf.avg_ts_l10, rdf.avg_ts_l10)

        # Average speed-up 50 ts
        su_t50_m, su_t50_lo, su_t50_up = compute_speed_up(su_t50_m, su_t50_lo, su_t50_up, ldf.avg_ts_l50, rdf.avg_ts_l50)

        # Total return
        tr_m, tr_lo, tr_up = compute_speed_up(tr_m, tr_lo, tr_up, ldf.total_return, rdf.total_return, rmax_m_lrmax=False)

        # Discounted return
        dr_m, dr_lo, dr_up = compute_speed_up(dr_m, dr_lo, dr_up, ldf.discounted_return, rdf.discounted_return, rmax_m_lrmax=False)

    label_data_dict = {
        r'$\rho_{Lip}$ (\% use Lipschitz bound)': (rlbu_m, rlbu_lo, rlbu_up),
        # r'\% time-steps to convergence gained': (su_m, su_lo, su_up),
        # r'\% average speed-up 2': (su_t2_m, su_t2_lo, su_t2_up),
        # r'\% average speed-up 5': (su_t5_m, su_t5_lo, su_t5_up),
        # r'\% average speed-up 10': (su_t10_m, su_t10_lo, su_t10_up),
        r'$\rho_{Speed\text{-}up}$ (\% convergence speed-up)': (su_t50_m, su_t50_lo, su_t50_up),  # r'\% average speed-up 50': (su_t50_m, su_t50_lo, su_t50_up),
        r'$\rho_{Return}$ (\% total return gain)': (tr_m, tr_lo, tr_up),
        # r'\% discounted return gained': (dr_m, dr_lo, dr_up)
    }

    '''
    for key, val in label_data_dict.items():
        print(key)
        print(val[0])
        print(val[1])
        print(val[2])
    exit()
    '''

    my_plot_bound_use(
        path=path,
        pdf_name='bounds_comparison',
        x=x,
        label_data_dict=label_data_dict,
        open_plot=open_plot
    )


def my_plot_bound_use(
        path,
        pdf_name,
        x,
        label_data_dict,
        open_plot=False,
        plot_max_bar=True,
        latex_rendering=True
):
    # LaTeX rendering
    if latex_rendering:
        rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

    colors = [[shade / 255.0 for shade in rgb] for rgb in RGB_COLORS_LST]
    markers = ['o', 's', 'D', '^', '*', 'x', 'p', '+', 'v', '|']

    x_margin = 0.05
    y_margin = 5.
    plt.xlim(max(x) + x_margin, min(x) - x_margin)  # decreasing upper-bound
    plt.ylim(0. - y_margin, 160. + y_margin)

    if plot_max_bar:
        plt.plot([max(x) + x_margin, min(x) - x_margin], [100., 100.], linestyle='-', color='black', linewidth=2)
        plt.gca().get_yticklabels()[6].set_weight('black')  # .set_color('red')
        plt.gca().get_yticklabels()[6].set_fontsize(20)
        # plt.gca().get_yticklabels()[6].set_bbox(dict(facecolor="white", alpha=1))

    i = 0
    for key, value in label_data_dict.items():
        plt.plot(x, value[0], markers[i], linestyle='-', color=colors[i], label=key)
        plt.fill_between(x, value[1], value[2], color=colors[i], alpha=0.3)
        i += 1

    if latex_rendering:
        plt.xlabel(r'Prior knowledge (known upper-bound on $\max_{s, a} = D_{s a} ( M \| \bar{M} )$)')
    else:
        plt.xlabel(r'Prior knowledge')
    plt.ylabel(r'\%')
    plt.legend(loc='best')
    # plt.title('')
    plt.grid(True, linestyle='--')
    plt.subplots_adjust(bottom=0.15)

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
    print('Running experiment, id =', index)
    p = PARAM[index]

    # Parameters
    gamma = 0.9
    n_instances = 1
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
    exp_id = 3  # default

    bounds_comparison_experiment(exp_id, do_run=False, do_plot=True, open_plot=True)
    exit()

    if len(sys.argv) == 2:
        exp_id = int(sys.argv[1])

    if exp_id == -1:  # run everything
        for _exp_id in range(len(PARAM)):
            bounds_comparison_experiment(_exp_id, do_run=True, do_plot=True, open_plot=False)
    else:
        bounds_comparison_experiment(exp_id, do_run=True, do_plot=True, open_plot=True)

