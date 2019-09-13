import sys
import os
import pandas
import numpy as np
from cycler import cycler
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib.ticker import MaxNLocator

from llrl.utils.utils import mean_confidence_interval
from llrl.utils.save import csv_path_from_agent

COLOR_SHIFT = 0

'''
color_ls = [
    [118, 167, 125],
    [102, 120, 173],
    [198, 113, 113],
    [94, 94, 94],
    [169, 193, 213],
    [230, 169, 132],
    [192, 197, 182],
    [210, 180, 226],
    [167, 167, 125],
    [125, 167, 125]
]
'''
color_ls = [
    [153, 194, 255],

    [159, 198, 177],
    [128, 179, 151],
    [96, 160, 126],

    [90, 90, 90],

    [255, 166, 77],
    [255, 102, 102]
]


def lifelong_plot(
        agents,
        path,
        n_tasks,
        n_episodes,
        confidence,
        open_plot,
        plot_title,
        plot_legend=0,
        legend_at_bottom=False,
        episodes_moving_average=False,
        episodes_ma_width=10,
        tasks_moving_average=False,
        tasks_ma_width=10,
        latex_rendering=False
):
    """
    Special plot routine for lifelong experiments.
    :param agents: (list)
    :param path: (str)
    :param n_tasks: (int)
    :param n_episodes: (int)
    :param confidence: (float)
    :param open_plot: (bool)
    :param plot_title: (bool)
    :param plot_legend: (int) takes several possible values:
        0: no legend
        1: only plot the legend for graphs displaying results w.r.t. episodes
        2: only plot the legend for graphs displaying results w.r.t. tasks
        3: legend for all
    :param legend_at_bottom: (bool)
    :param episodes_moving_average: (bool)
    :param episodes_ma_width: (int)
    :param tasks_moving_average: (bool)
    :param tasks_ma_width: (int)
    :param latex_rendering: (bool)
    :return: None
    """
    dfs = []
    for agent in agents:
        agent_path = csv_path_from_agent(path, agent)
        dfs.append(pandas.read_csv(agent_path))

    tre, tre_lo, tre_up = [], [], []
    dre, dre_lo, dre_up = [], [], []
    trt, trt_lo, trt_up = [], [], []
    drt, drt_lo, drt_up = [], [], []
    for i in range(len(agents)):
        tre_i, tre_lo_i, tre_up_i = [], [], []
        dre_i, dre_lo_i, dre_up_i = [], [], []
        for j in range(1, n_episodes + 1):
            df = dfs[i].loc[dfs[i]['episode'] == j]
            tre_mci_j = mean_confidence_interval(df['return'], confidence)
            tre_i.append(tre_mci_j[0])
            tre_lo_i.append(tre_mci_j[1])
            tre_up_i.append(tre_mci_j[2])
            dre_mci_j = mean_confidence_interval(df['discounted_return'], confidence)
            dre_i.append(dre_mci_j[0])
            dre_lo_i.append(dre_mci_j[1])
            dre_up_i.append(dre_mci_j[2])
        tre.append(tre_i)
        tre_lo.append(tre_lo_i)
        tre_up.append(tre_up_i)
        dre.append(dre_i)
        dre_lo.append(dre_lo_i)
        dre_up.append(dre_up_i)

        trt_i, trt_lo_i, trt_up_i = [], [], []
        drt_i, drt_lo_i, drt_up_i = [], [], []
        for j in range(1, n_tasks + 1):
            df = dfs[i].loc[dfs[i]['task'] == j]
            trt_mci_j = mean_confidence_interval(df['return'], confidence)
            trt_i.append(trt_mci_j[0])
            trt_lo_i.append(trt_mci_j[1])
            trt_up_i.append(trt_mci_j[2])
            drt_mci_j = mean_confidence_interval(df['discounted_return'], confidence)
            drt_i.append(drt_mci_j[0])
            drt_lo_i.append(drt_mci_j[1])
            drt_up_i.append(drt_mci_j[2])
        trt.append(trt_i)
        trt_lo.append(trt_lo_i)
        trt_up.append(trt_up_i)
        drt.append(drt_i)
        drt_lo.append(drt_lo_i)
        drt_up.append(drt_up_i)

    x_e = np.array(range(1, n_episodes + 1))
    x_t = np.array(range(1, n_tasks + 1))
    x_label_e = r'Episode number'
    x_label_t = r'Task number'

    # Plots w.r.t. episodes
    plot_legend = True if plot_legend == 1 or plot_legend == 3 else False
    plot(path, pdf_name='return_vs_episode', agents=agents, x=x_e, y=tre, y_lo=tre_lo, y_up=tre_up,
         x_label=x_label_e, y_label=r'Average Return', title_prefix=r'Average Return: ', open_plot=open_plot,
         plot_title=plot_title, plot_legend=plot_legend, legend_at_bottom=legend_at_bottom,
         moving_average=episodes_moving_average, ma_width=episodes_ma_width, latex_rendering=latex_rendering,
         x_cut=None)
    plot(path, pdf_name='discounted_return_vs_episode', agents=agents, x=x_e, y=dre, y_lo=dre_lo, y_up=dre_up,
         x_label=x_label_e, y_label=r'Average Discounted Return', title_prefix=r'Average Discounted Return: ',
         open_plot=open_plot, plot_title=plot_title, plot_legend=plot_legend, legend_at_bottom=legend_at_bottom,
         moving_average=episodes_moving_average, ma_width=episodes_ma_width, latex_rendering=latex_rendering,
         x_cut=None)

    # Plots w.r.t. tasks
    plot_legend = True if plot_legend == 2 or plot_legend == 3 else False
    plot(path, pdf_name='return_vs_task', agents=agents, x=x_t, y=trt, y_lo=trt_lo, y_up=trt_up,
         x_label=x_label_t, y_label=r'Average Return', title_prefix=r'Average Return: ', open_plot=open_plot,
         plot_title=plot_title, plot_legend=plot_legend, legend_at_bottom=legend_at_bottom,
         moving_average=tasks_moving_average, ma_width=tasks_ma_width, latex_rendering=latex_rendering)
    plot(path, pdf_name='discounted_return_vs_task', agents=agents, x=x_t, y=drt, y_lo=drt_lo, y_up=drt_up,
         x_label=x_label_t, y_label=r'Average Discounted Return', title_prefix=r'Average Discounted Return: ',
         open_plot=open_plot, plot_title=plot_title, plot_legend=plot_legend, legend_at_bottom=legend_at_bottom,
         moving_average=tasks_moving_average, ma_width=tasks_ma_width, latex_rendering=latex_rendering)


def compute_moving_average(w, x, y, y_lo=None, y_up=None):
    """
    Compute the moving average.
    :param w: (int) width
    :param x: (array-like)
    :param y: (array-like)
    :param y_lo: (array-like)
    :param y_up: (array-like)
    :return:
    """
    assert w > 1, 'Error: moving average width must be > 1: w = {}'.format(w)
    assert len(x) == len(y), 'Error: x and y vector should have the same length: len(x) = {}, len(y) = {}'.format(
        len(x), len(y))

    n = len(x)
    w_2 = int(w / 2)
    x, y = np.array(x), np.array(y)
    x_new, y_new = [], []
    y_lo_new = None if y_lo is None else []
    y_up_new = None if y_up is None else []
    indexes = list(range(w_2, n, w))

    for i in indexes:
        x_new.append(np.mean(x[i - w_2: i + w_2 - 1]))
        y_new.append(np.mean(y[i - w_2: i + w_2 - 1]))
        if y_lo is not None:
            y_lo_new.append(np.mean(y_lo[i - w_2: i + w_2 - 1]))
        if y_up is not None:
            y_up_new.append(np.mean(y_up[i - w_2: i + w_2 - 1]))

    x_new = np.insert(x_new, 0, x[0])
    x_new = np.append(x_new, x[-1])

    y_new = np.insert(y_new, 0, np.mean(y[0:w_2]))
    y_new = np.append(y_new, np.mean(y[-w_2:]))

    if y_lo is not None:
        y_lo_new = np.insert(y_lo_new, 0, np.mean(y_lo[0:w_2]))
        y_lo_new = np.append(y_lo_new, np.mean(y_lo[-w_2:]))

    if y_up is not None:
        y_up_new = np.insert(y_up_new, 0, np.mean(y_up[0:w_2]))
        y_up_new = np.append(y_up_new, np.mean(y_up[-w_2:]))

    return x_new, y_new, y_lo_new, y_up_new


def plot(
        path,
        pdf_name,
        agents,
        x,
        y,
        y_lo,
        y_up,
        x_label,
        y_label,
        title_prefix,
        x_cut=None,
        open_plot=True,
        plot_title=True,
        plot_markers=True,
        plot_legend=True,
        legend_at_bottom=False,
        moving_average=True,
        ma_width=10,
        latex_rendering=False
):
    """
    Tweaked version of simple_rl.utils.chart_utils.plot
    Method made less specific, no specification of the type of data.
    :param path: (str) experiment path
    :param pdf_name: (str)
    :param agents: (list) list of agents
    :param x: (list) x axis data
    :param y: (list) list of array-like containing the x data for each agent
    :param y_lo: (list) list of array-like containing the lower bound on the confidence interval of the x data
    :param y_up: (list) list of array-like containing the upper bound on the confidence interval of the x data
    :param x_label: (str)
    :param y_label: (str)
    :param title_prefix: (str)
    :param x_cut: (int) cut the x_axis, does nothing if set to None
    :param open_plot: (Bool)
    :param plot_title: (Bool)
    :param plot_markers: (Bool)
    :param plot_legend: (Bool)
    :param legend_at_bottom: (Bool)
    :param moving_average: (Bool)
    :param ma_width: (int)
    :param latex_rendering: (Bool)
    :return: None
    """
    # x-cut
    if x_cut is not None:
        x = x[:x_cut]
        for i in range(len(agents)):
            y[i] = y[i][:x_cut]
            y_lo[i] = y_lo[i][:x_cut]
            y_up[i] = y_up[i][:x_cut]

    # LaTeX rendering
    if latex_rendering:
        rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Set markers and colors
    markers = ['o', 's', 'D', '^', '*', 'x', 'p', '+', 'v', '|']
    colors = [[shade / 255.0 for shade in rgb] for rgb in color_ls]
    colors = colors[COLOR_SHIFT:] + colors[:COLOR_SHIFT]
    ax.set_prop_cycle(cycler('color', colors))

    for i in range(len(agents)):
        label_i = _format_label(str(agents[i]), latex_rendering)
        if moving_average:
            _x, y[i], y_lo[i], y_up[i] = compute_moving_average(ma_width, x, y[i], y_lo[i], y_up[i])
        else:
            _x = x
        if y_lo is not None and y_up is not None:
            plt.fill_between(_x, y_lo[i], y_up[i], alpha=0.25, facecolor=colors[i], edgecolor=colors[i])
        if plot_markers:
            plt.plot(_x, y[i], '-o', label=label_i, marker=markers[i])
        else:
            plt.plot(_x, y[i], label=label_i)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # plt.ylim(bottom=0)

    if plot_legend:
        if legend_at_bottom:
            # Shrink current axis's height by p% on the bottom
            p = 0.4
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * p, box.width, box.height * (1.0 - p)])
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=False, shadow=False, ncol=2)
        else:
            plt.legend(loc='best')

    plt.grid(True, linestyle='--')

    exp_dir_split_list = path.split("/")
    if 'results' in exp_dir_split_list:
        exp_name = exp_dir_split_list[exp_dir_split_list.index('results') + 1]
    else:
        exp_name = exp_dir_split_list[0]
    if plot_title:
        plt_title = _format_title(title_prefix + exp_name)
        plt.title(plt_title)

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


def _format_title(title):
    title = title.replace("_", " ")
    title = title.replace("-", " ")
    if len(title.split(" ")) > 1:
        return " ".join([w[0].upper() + w[1:] for w in title.strip().split(" ")])


def _format_label(label, latex_rendering):
    if latex_rendering:
        label = label.replace('Dmax=', r'$D_{\max} =$ ')
    return label
