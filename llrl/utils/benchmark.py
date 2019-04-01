"""
Benchmark functions for multi-core / multi-threading processes
"""

import multiprocessing
import time


def multi_cores(atomic_function, n_cores, core_number_as_arg=True, verbose=True):
    """
    Execute the input function on the given number of cores / threads.
    :param atomic_function: function to be executed taking only two arguments: (core_number, thread_number)
    :param n_cores: (int) number of cores
    :param core_number_as_arg: (bool)
    :param verbose: (bool)
    :return: None
    """
    if verbose:
        print('Executing', atomic_function, '( number of cores:', n_cores, ')')

    start = time.clock()

    jobs = []
    for i in range(n_cores):
        if core_number_as_arg:
            process = multiprocessing.Process(target=atomic_function, args=(i + 1,))
        else:
            process = multiprocessing.Process(target=atomic_function)
        jobs.append(process)

    for job in jobs:
        job.start()

    for job in jobs:
        job.join()

    end = time.clock()

    if verbose:
        print('Successfully executed', atomic_function, '( number of cores:', n_cores, ')')
        print('Execution time:', str(round(end - start, 3)), '(s)')
