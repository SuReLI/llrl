import csv
import pandas as pd


def lifelong_save(path, agent, data, n_instance, is_first_save):
    """
    Save according to a specific data structure designed for lifelong RL experiments.
    :param path: (str)
    :param agent: agent object
    :param data: (dictionary)
    :return: None
    """
    full_path = path + '/results-' + agent.get_name() + '.csv'
    if is_first_save:
        names = ['instance', 'task', 'episode', 'return', 'discounted_return']
        csv_write(names, full_path, 'w')
    # TODO here


def save_return_per_episode(
        path,
        agents,
        avg_return_per_episode_per_agent,
        is_tracked_value_discounted
):
    # Set names
    labels = [
        'episode_number', 'average_discounted_return', 'average_discounted_return_lo', 'average_discounted_return_up'
    ] if is_tracked_value_discounted else [
        'episode_number', 'average_return', 'average_return_lo', 'average_return_up'
    ]
    file_name = 'average_discounted_return_per_episode' if is_tracked_value_discounted else 'average_return_per_episode'

    # Save
    n_episodes = len(avg_return_per_episode_per_agent[0])
    x = range(n_episodes)
    data = []
    for agent in range(len(agents)):
        data.append([])
        for episode in range(n_episodes):
            data[-1].append([
                x[episode],
                avg_return_per_episode_per_agent[agent][episode][0],
                avg_return_per_episode_per_agent[agent][episode][1],
                avg_return_per_episode_per_agent[agent][episode][2]
            ])
    save_agents(path, csv_name=file_name, agents=agents, data=data, labels=labels)


def get_csv_path(path, csv_name, agent):
    return path + '/' + csv_name + '-' + agent.get_name() + '.csv'


def save_agents(path, csv_name, agents, data, labels):
    """
    Write the specified data at the specified path in csv format.
    :param path: (str) root path
    :param csv_name: (str)
    :param agents: (list)
    :param data: (list) data[agent][row][item]
    :param labels: (list)
    :return: None
    """
    for agent in range(len(agents)):
        csv_path = get_csv_path(path, csv_name, agents[agent])
        csv_write(labels, csv_path, 'w')
        for r in range(len(data[agent])):
            row = []
            for item in range(len(data[agent][r])):
                row.append(data[agent][r][item])
            csv_write(row, csv_path, 'a')


def open_agents(path, csv_name, agents):
    """
    Opposite of save method.
    :param path: (str) root path
    :param csv_name: (str)
    :param agents: (list)
    :param data: (list) data[agent][row][item]
    :param labels: (list)
    :return: None
    """
    data_frames = []
    for agent in range(len(agents)):
        csv_path = get_csv_path(path, csv_name, agents[agent])
        data_frames.append(pd.read_csv(csv_path))
    return data_frames


def csv_write(row, path, mode):
    """
    Write a row into a csv.
    :param row: (array-like) written row, array-like whose elements are separated in the output file.
    :param path: (str) path to the edited csv
    :param mode: (str) mode for writing: 'w' override, 'a' append
    :return: None
    """
    with open(path, mode) as csv_file:
        w = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        w.writerow(row)
