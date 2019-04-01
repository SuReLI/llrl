"""
Simple example for multi-processing use.
"""

import time

from llrl.utils.save import csv_write
from llrl.utils.benchmark import multi_cores


def my_atomic_function(core_number):
    """
    Atomic function to be executed by each core / thread.
    :param core_number: (int) core number identifier
    :return: None
    """
    path = 'tmp/benchmark-example-' + str(core_number) + '.csv'
    for i in range(3):
        time.sleep(1)
        csv_write([i], path, 'w')


if __name__ == "__main__":
    n_cores = 4
    multi_cores(my_atomic_function, n_cores)
