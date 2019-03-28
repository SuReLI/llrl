"""
Lifelong RL experiment in constant transition function setting
"""

from llrl.agents.lrmax_ct import LRMaxCT
from simple_rl.run_experiments import run_agents_lifelong
from simple_rl.utils import make_mdp


def experiment():
    mdp_distribution = make_mdp.make_mdp_distr(mdp_class="four_room")


if __name__ == '__main__':
    experiment()
