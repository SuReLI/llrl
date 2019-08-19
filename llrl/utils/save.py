import csv


def csv_path_from_agent(root_path, agent):
    """
    Get the saving path from agent object and root path.
    :param root_path: (str)
    :param agent: (object)
    :return: (str)
    """
    return root_path + '/results-' + agent.get_name() + '.csv'


def lifelong_save(path, agent, data, instance_number, is_first_save):
    """
    Save according to a specific data structure designed for lifelong RL experiments.
    :param path: (str)
    :param agent: agent object
    :param data: (dictionary)
    :param instance_number: (int)
    :param is_first_save: (bool)
    :return: None
    """
    full_path = csv_path_from_agent(path, agent)
    n_tasks = len(data['returns_per_tasks'])
    n_episodes = len(data['returns_per_tasks'][0])

    if is_first_save:
        names = ['instance', 'task', 'episode', 'return', 'discounted_return']
        csv_write(names, full_path, 'w')

    for i in range(n_tasks):
        for j in range(n_episodes):
            row = [str(instance_number), str(i + 1), str(j + 1), data['returns_per_tasks'][i][j],
                   data['discounted_returns_per_tasks'][i][j]]
            csv_write(row, full_path, 'a')


# TODO remove
'''
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
'''


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
