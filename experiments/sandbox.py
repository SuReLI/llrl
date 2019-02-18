import llrl.utils.distribution as dist
import numpy as np


def test():
    inf = 1e99
    d = np.array(
        [
            [0.0, 0.9],
            [0.9, 1.0]
        ]
    )
    n = d.shape[0]
    uniform = (1.0 / float(n)) * np.ones(shape=n, dtype=float)
    distance, match = dist.wass_primal(uniform, uniform, d)
    print('distance :', distance)
    print('match    :', match)


if __name__ == "__main__":
    test()
