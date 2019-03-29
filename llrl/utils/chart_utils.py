import sys
import os
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib.ticker import MaxNLocator

from simple_rl.utils.chart_utils import _format_title


def plot(path, pdf_name, agents, x, y, y_lo, y_up, x_label, y_label, title_prefix, open_plot=True):
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
    :param open_plot: (Bool)
    :return: None
    """
    # LaTeX rendering
    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    for i in range(len(agents)):
        plt.plot(x, y[i], '-o', label=agents[i])
        if y_lo is not None and y_up is not None:
            plt.fill_between(x, y_lo[i], y_up[i], alpha=0.2)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc='best')
    plt.grid(True, linestyle='--')
    exp_dir_split_list = path.split("/")
    if 'results' in exp_dir_split_list:
        exp_name = exp_dir_split_list[exp_dir_split_list.index('results') + 1]
    else:
        exp_name = exp_dir_split_list[0]
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