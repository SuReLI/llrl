import csv


def save(path, csv_name, agents, data, labels):
    """
    Write the specified data at the specified path in csv format.
    :param path: (str) root path
    :param csv_name: (str)
    :param agents: (list)
    :param data: (list) data[agent][row][item]
    :param labels: (list)
    :return: None
    """
    print()
    print()
    print('SAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAVE')
    for agent in range(len(agents)):
        csv_path = path + '/' + csv_name + '-' + agents[agent].get_name() + '.csv'
        csv_write(labels, csv_path, 'w')
        print(csv_path)
        print(labels)
        for r in range(len(data[agent])):
            row = []
            for item in range(len(data[agent][r])):
                row.append(data[agent][r][item])
            csv_write(row, csv_path, 'a')
            print(row)


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
