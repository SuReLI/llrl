from llrl.agents.rmax import RMax
from llrl.agents.lrmax import LRMax
from llrl.utils.env_handler import make_env_distribution
from llrl.experiments_maker import run_agents_lifelong


GAMMA = .9


def example():
    n_env = 5
    env_distribution = make_env_distribution(env_class='corridor', n_env=n_env, gamma=GAMMA, w=20, h=20)
    actions = env_distribution.get_actions()

    m = 1
    max_mem = None
    p_min = 1. / float(n_env)
    lrmax = LRMax(
        actions=actions, gamma=GAMMA, count_threshold=m, max_memory_size=max_mem,
        prior=None, min_sampling_probability=p_min, delta=0.4
    )
    rmax = RMax(actions=actions, gamma=GAMMA, count_threshold=m)

    run_agents_lifelong(
        [lrmax, rmax], env_distribution, samples=10, episodes=10, steps=100, reset_at_terminal=False,
        open_plot=True, cumulative_plot=False, is_tracked_value_discounted=True, plot_only=False
    )


if __name__ == '__main__':
    # np.random.seed(1993)
    example()
