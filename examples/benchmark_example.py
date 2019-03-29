"""
Simple example for use of the llrl.utils.benchmark functions.
"""

import time

from llrl.utils.save import csv_write


def my_atomic_function(core_number, thread_number):
    """
    Atomic function to be executed by each core / thread.
    :param core_number: (int) core number identifier
    :param thread_number: (int) thread number identifier
    :return: None
    """
    path = 'tmp/benchmark-example-' + core_number + '-' + thread_number + '.csv'
    csv_write(['counter'], path, 'w')
    for i in range(100):
        time.sleep(1)
        csv_write([i], path, 'a')

