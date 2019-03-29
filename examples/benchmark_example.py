"""
Simple example for use of the llrl.utils.benchmark functions.
"""

import time

from llrl.utils.save import csv_write
from llrl.utils.benchmark import multi


def my_atomic_function(core_number, thread_number):
    """
    Atomic function to be executed by each core / thread.
    :param core_number: (int) core number identifier
    :param thread_number: (int) thread number identifier
    :return: None
    """
    path = 'tmp/benchmark-example-' + str(core_number) + '-' + str(thread_number) + '.csv'
    csv_write(['counter'], path, 'w')
    for i in range(100):
        time.sleep(1)
        csv_write([i], path, 'a')


if __name__ == "__main__":
    n_cores = 4
    n_threads = 1

    multi(my_atomic_function, n_cores, n_threads)
