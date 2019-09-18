import numpy as np
import scipy.stats
from math import isclose


def mean_confidence_interval(data, confidence=0.95):
    """
    Compute the mean and confidence interval of the the input data array-like.
    :param data: (array-like)
    :param confidence: probability of the mean to lie in the interval
    :return: (tuple) mean, interval upper-endpoint, interval lower-endpoint
    """
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


def list_average(lst):
    return sum(lst) / len(lst)


def avg_last_elts(lst, n):
    if len(lst) < n:
        print('Warning: in method avg_last_elts, list length (', len(lst),
              ') is smaller than number of elements (', n, ')')
        return list_average(lst)
    else:
        if n > 3:  # TODO remove
            return list_average(lst[-n:-2])
        else:
            return list_average(lst[-n:])


def close(a, b, r=13):
    return isclose(round(a,r), round(b,r), rel_tol=1e-12, abs_tol=0.0)


def closevec(u, v, r=13):
    assert len(u) == len(v), 'Error: vectors have different lengths: len(u)={} len(v)={}'.format(len(u), len(v))
    for i in range(len(u)):
        if not close(u[i], v[i], r):
            return False
    return True


def are_coeff_equal(v):
    return bool(np.prod(list(v[i] == v[i+1] for i in range(len(v)-1)), axis=0))


def are_coeff_close(v):
    return bool(np.prod(list(close(v[i],v[i+1]) for i in range(len(v)-1)), axis=0))


def assert_types(p, types_list):
    """
    Assert that the types of the elements of p match those of the types_list
    """
    assert len(p) == len(types_list), 'Error: expected {} parameters received {}'.format(len(types_list), len(p))
    for i in range(len(p)):
        assert type(p[i]) == types_list[i], 'Error: wrong type, expected {}, received {}'.format(types_list[i], type(p[i]))


def amax(v):
    """
    Return the higher value and its index given an array of values.
    """
    vmax, index = v[0], 0
    for i in range(1, len(v)):
        if v[i] > vmax:
            vmax = v[i]
            index = i
    return vmax, index
