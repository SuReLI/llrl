import sys
import os
import pandas
from cycler import cycler
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib.ticker import MaxNLocator

from llrl.utils.utils import mean_confidence_interval
from llrl.utils.save import csv_path_from_agent

COLOR_SHIFT = 0

color_ls = [
    [118, 167, 125], [102, 120, 173], [198, 113, 113], [94, 94, 94], [169, 193, 213],
    [230, 169, 132], [192, 197, 182], [210, 180, 226], [167, 167, 125], [125, 167, 125]
]


def lifelong_plot(agents, path):
    dfs = []
    for agent in agents:
        agent_path = csv_path_from_agent(path, agent)
        dfs.append(pandas.read_csv(agent_path))

    print(dfs[0])

    '''
    # Set names
    labels = [
        'episode_number', 'average_discounted_return', 'average_discounted_return_lo', 'average_discounted_return_up'
    ] if is_tracked_value_discounted else [
        'episode_number', 'average_return', 'average_return_lo', 'average_return_up'
    ]
    file_name = 'average_discounted_return_per_episode' if is_tracked_value_discounted else 'average_return_per_episode'
    x_label = r'Episode Number'
    y_label = r'Average Discounted Return' if is_tracked_value_discounted else r'Average Return'
    title_prefix = r'Average Discounted Return: ' if is_tracked_value_discounted else r'Average Return: '

    # Open data
    data_frames = open_agents(path, csv_name=file_name, agents=agents)

    # Plot
    n_episodes = len(data_frames[0][labels[0]])
    x = range(n_episodes)
    returns = []
    returns_lo = []
    returns_up = []
    for df in data_frames:
        returns.append(df[labels[1]][0:n_episodes])
        returns_lo.append(df[labels[2]][0:n_episodes])
        returns_up.append(df[labels[3]][0:n_episodes])
    plot(
        path, pdf_name=file_name, agents=agents, x=x, y=returns, y_lo=returns_lo, y_up=returns_up,
        x_label=x_label, y_label=y_label, title_prefix=title_prefix, open_plot=open_plot, plot_title=plot_title
    )
    '''


def plot(path, pdf_name, agents, x, y, y_lo, y_up, x_label, y_label, title_prefix, open_plot=True, plot_title=True):
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
    :param plot_title: (Bool)
    :return: None
    """
    # LaTeX rendering
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
        if y_lo is not None and y_up is not None:
            plt.fill_between(x, y_lo[i], y_up[i], alpha=0.25, facecolor=colors[i], edgecolor=colors[i])
        plt.plot(x, y[i], '-o', label=agents[i], marker=markers[i])

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc='best')
    plt.grid(True)  # , linestyle='--')
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


def _format_title(plot_title):
    plot_title = plot_title.replace("_", " ")
    plot_title = plot_title.replace("-", " ")
    if len(plot_title.split(" ")) > 1:
        return " ".join([w[0].upper() + w[1:] for w in plot_title.strip().split(" ")])
