"""
Comparison of the bounds provided by R-MAX and Lipschitz-R-Max.
"""

from llrl.envs.twostates import TwoStates
from llrl.agents.lrmax import LRMax
from llrl.agents.rmax_vi import RMaxVI
from simple_rl.agents import RandomAgent
from simple_rl.run_experiments import run_agents_on_mdp
from simple_rl.tasks import GridWorldMDP


def main():
    # MDP
    size = 5
    mdp1 = GridWorldMDP(width=size, height=size, init_loc=(1, 1), goal_locs=[(size, size)])
    # mdp1 = TwoStates(rewards=(0., 1.), proba=(0., 0.))
    # mdp2 = TwoStates(rewards=(0., 1.), proba=(.9, .9))

    # Agents
    rand_agent = RandomAgent(actions=mdp1.get_actions())
    rmax_vi_agent = RMaxVI(actions=mdp1.get_actions(), gamma=.9, count_threshold=2)
    lrmax_agent = LRMax(actions=mdp1.get_actions(), gamma=.9, count_threshold=2)

    # Run
    run_agents_on_mdp([lrmax_agent, rmax_vi_agent, rand_agent], mdp1, instances=1, episodes=100, steps=30,
                      reset_at_terminal=True, verbose=False, open_plot=False)

    run_agents_on_mdp([lrmax_agent, rmax_vi_agent, rand_agent], mdp1, instances=1, episodes=100, steps=30,
                      reset_at_terminal=True, verbose=False, open_plot=False)


if __name__ == "__main__":
    main()
