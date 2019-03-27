"""
Comparison of the bounds provided by R-MAX and Lipschitz-R-Max.

Setting:
- Using two grid-world MDPs with same transition functions and different reward functions while reaching goal;
- Learning on the first MDP and transferring the Lipschitz bound to the second one;
- Plotting percentage of bound use vs amount of prior knowledge.
"""

import numpy as np

from llrl.envs.gridworld import GridWorld
from llrl.agents.lrmax_constant_transitions_testing import LRMaxCTTesting
from simple_rl.run_experiments import run_agents_on_mdp


SAVE_PATH = "results/bounds_comparison_results.csv"
PRIOR = np.linspace(0., 1., num=11)


def main():
    # MDP
    size = 5
    mdp1 = GridWorld(width=size, height=size, init_loc=(1, 1), goal_locs=[(size, size)], goal_reward=1.0)
    mdp2 = GridWorld(width=size, height=size, init_loc=(1, 1), goal_locs=[(size, size)], goal_reward=0.8)

    for prior in PRIOR:
        lrmaxct = LRMaxCTTesting(actions=mdp1.get_actions(), gamma=.9, count_threshold=1, delta_r=prior, path=SAVE_PATH)

        # Run twice
        run_agents_on_mdp([lrmaxct], mdp1, instances=1, episodes=100, steps=30,
                          reset_at_terminal=True, verbose=False, open_plot=False)

        run_agents_on_mdp([lrmaxct], mdp2, instances=1, episodes=100, steps=30,
                          reset_at_terminal=True, verbose=False, open_plot=False)


if __name__ == "__main__":
    main()
