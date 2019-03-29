"""
Benchmark functions for multi-core / multi-threading processes
"""

import multiprocessing


def multi(atomic_function, n_cores, n_threads, verbose=True):
    """
    Execute the input function on the given number of cores / threads.
    :param atomic_function: function to be executed taking only two arguments: (core_number, thread_number)
    :param n_cores: (int) number of cores
    :param n_threads: (int) number of threads
    :return: None
    """
    jobs = []
    for i in range(n_cores):
        j = 0  # TODO thread number
        process = multiprocessing.Process(target=atomic_function, args=(i + 1, j + 1))
        jobs.append(process)

    for job in jobs:
        job.start()

    for job in jobs:
        job.join()

    if verbose:
        print('Successfully executed', atomic_function, '( n_cores:', n_cores, 'n_threads:0', n_threads, ')')
