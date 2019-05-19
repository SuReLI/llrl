"""
Negative transfer example using Song et al. [2016] transfer method.
"""


from llrl.envs.n_states import NStates


def weighted_transfer():
    ns = 2
    env1 = NStates(ns)
    # env2 = NStates(ns)


def state_transfer():
    print('TODO')


if __name__ == "__main__":
    weighted_transfer()
    state_transfer()
