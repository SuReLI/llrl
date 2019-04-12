import sys
import os
import pandas as pd
from cycler import cycler
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib.ticker import MaxNLocator

from llrl.utils.utils import mean_confidence_interval
from llrl.utils.save import csv_write
from llrl.utils.chart_utils import color_ls
from llrl.utils.chart_utils import COLOR_SHIFT

MARKERS = ['o', 's', 'D', '^', '*', 'x', 'p', '+', 'v', '|']
COLORS = [[shade / 255.0 for shade in rgb] for rgb in color_ls]
COLORS = COLORS[COLOR_SHIFT:] + COLORS[:COLOR_SHIFT]


def get_path_computation_number(root_path, agent_name):
    return root_path + agent_name + 'computation_number.csv'


def get_path_time_step(root_path, agent_name):
    return root_path + agent_name + 'time_step.csv'


def save_result(results, root_path, name):
    path = get_path_computation_number(root_path, name)
    csv_write(['prior_use_ratio_mean', 'prior_use_ratio_lo', 'prior_use_ratio_up'], path, mode='w')

    length = max([len(r) for r in results])
    for i in range(length):
        data_i = []
        for r in results:
            if len(r) > i:
                data_i.append(r[i][1])
        mean, lo, up = mean_confidence_interval(data_i)
        csv_write([mean, lo, up], path, mode='a')

    path = get_path_time_step(root_path, name)
    csv_write(['time_step', 'prior_use_ratio'], path, mode='w')
    for r in results:
        for row in r:
            csv_write([row[0], row[1]], path, mode='a')


def moving_average(x, y, window=2):
    x_ma, y_ma = [], []
    for i in range(window, len(x) + 1):
        x_ma.append(sum(x[i - window: i]) / float(window))
        y_ma.append(sum(y[i - window: i]) / float(window))
    return x_ma, y_ma


def plot_time_step_results(root_path, names, open_plot=True):
    # LaTeX rendering
    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_prop_cycle(cycler('color', COLORS))

    for i in range(len(names)):
        df = pd.read_csv(get_path_time_step(root_path, names[i]))
        time_step = df.time_step
        prior_use_ratio = df.prior_use_ratio

        time_step, prior_use_ratio = zip(*sorted(zip(time_step, prior_use_ratio)))
        x_ma, ma = moving_average(time_step, prior_use_ratio)

        plt.plot(x_ma, ma, '-o', marker=None)
        plt.scatter(time_step, prior_use_ratio, label=names[i], marker=MARKERS[i])

    plt.xlim((0, 20000))
    plt.xlabel(r'Time Step')
    plt.ylabel(r'\% Prior Use')
    plt.legend(loc='best')
    plt.grid(True)
    # plt.title('')

    # Save
    plot_file_name = os.path.join(root_path + 'prior_use_vs_time_step.pdf')
    plt.savefig(plot_file_name, format='pdf')

    # Open
    if open_plot:
        open_prefix = 'gnome-' if sys.platform == 'linux' or sys.platform == 'linux2' else ''
        os.system(open_prefix + 'open ' + plot_file_name)

    # Clear and close
    plt.cla()
    plt.close()


def plot_computation_number_results(root_path, names, open_plot=True):
    # LaTeX rendering
    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_prop_cycle(cycler('color', COLORS))

    for i in range(len(names)):
        df = pd.read_csv(get_path_computation_number(root_path, names[i]))
        prior_use_ratio_mean = df.prior_use_ratio_mean
        prior_use_ratio_lo = df.prior_use_ratio_lo
        prior_use_ratio_up = df.prior_use_ratio_up

        x = range(len(prior_use_ratio_mean))

        plt.plot(x, prior_use_ratio_mean, '-o', label=names[i], marker=MARKERS[i])
        plt.fill_between(x, prior_use_ratio_lo, prior_use_ratio_up, alpha=.25, facecolor=COLORS[i], edgecolor=COLORS[i])

    plt.ylim((-6., 106.))
    plt.xlabel(r'Computation Number')
    plt.ylabel(r'\% Prior Use')
    plt.legend(loc='best')
    plt.grid(True)
    # plt.title('')

    # Save
    plot_file_name = os.path.join(root_path + 'prior_use_vs_computation_number.pdf')
    plt.savefig(plot_file_name, format='pdf')

    # Open
    if open_plot:
        open_prefix = 'gnome-' if sys.platform == 'linux' or sys.platform == 'linux2' else ''
        os.system(open_prefix + 'open ' + plot_file_name)

    # Clear and close
    plt.cla()
    plt.close()