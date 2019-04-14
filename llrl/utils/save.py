import csv
import pandas as pd


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
