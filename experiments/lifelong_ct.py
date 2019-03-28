"""
Lifelong RL experiment in constant transition function setting
"""

# from llrl.agents.lrmax_ct import LRMaxCT
from llrl.agents.rmax_vi import RMaxVI
from llrl.utils.env_handler import make_env_distribution
from simple_rl.run_experiments import run_agents_lifelong


GAMMA = .9
WIDTH = 4
HEIGHT = 4


def experiment():
    # Create environments distribution
    env_distribution = make_env_distribution(env_class='grid-world', n_env=10, gamma=GAMMA, w=WIDTH, h=HEIGHT)

    rmax = RMaxVI(actions=env_distribution.get_actions(), gamma=GAMMA, count_threshold=1)

    run_agents_lifelong(
        [rmax], env_distribution, samples=20, episodes=50, steps=100, reset_at_terminal=True, open_plot=True
    )


if __name__ == '__main__':
    experiment()
