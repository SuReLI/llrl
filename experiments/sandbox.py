import numpy as np

from llrl.agents.rmax import RMax
from llrl.agents.lrmax import LRMax
from llrl.utils.env_handler import make_env_distribution
from llrl.experiments_maker import run_agents_lifelong


GAMMA = .9


def example():
    n_env = 5
    env_distribution = make_env_distribution(env_class='grid-world', n_env=n_env, gamma=GAMMA, w=3, h=3)
    actions = env_distribution.get_actions()

    m = 100
    max_mem = None
    p_min = 1. / float(n_env)
    lrmax = LRMax(
        actions=actions, gamma=GAMMA, count_threshold=m, max_memory_size=max_mem,
        prior=None, min_sampling_probability=p_min, delta=0.4
    )

    run_agents_lifelong(
        [lrmax], env_distribution, samples=100, episodes=100, steps=1000, reset_at_terminal=False,
        open_plot=True, cumulative_plot=False, is_tracked_value_discounted=True, plot_only=False
    )


if __name__ == '__main__':
    np.random.seed(1993)
    example()
