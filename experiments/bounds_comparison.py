"""
Comparison of the bounds provided by R-MAX and Lipschitz-R-Max.
"""

from llrl.envs.twostates import TwoStates
from llrl.envs.gridworld import GridWorld
from llrl.agents.lrmax import LRMax
from llrl.agents.rmax_vi import RMaxVI
from llrl.agents.lrmax_constant_transitions import LRMaxCT
from simple_rl.agents import RandomAgent
from simple_rl.run_experiments import run_agents_on_mdp
from simple_rl.tasks import GridWorldMDP


def main():
    # MDP
    size = 5
    mdp1 = GridWorld(width=size, height=size, init_loc=(1, 1), goal_locs=[(size, size)], goal_reward=1.0)
    mdp2 = GridWorld(width=size, height=size, init_loc=(1, 1), goal_locs=[(size, size)], goal_reward=1.0)
    # mdp1 = TwoStates(rewards=(0., 1.), proba=(0., 0.))
    # mdp2 = TwoStates(rewards=(0., 1.), proba=(.9, .9))

    # Agents
    rand = RandomAgent(actions=mdp1.get_actions())
    rmax = RMaxVI(actions=mdp1.get_actions(), gamma=.9, count_threshold=1)
    lrmax = LRMax(actions=mdp1.get_actions(), gamma=.9, count_threshold=1)
    lrmaxct = LRMaxCT(actions=mdp1.get_actions(), gamma=.9, count_threshold=1, delta_r=0.)

    agent_pool = [rmax, lrmaxct, rand]  # The agents you want to test

    # Run
    run_agents_on_mdp(agent_pool, mdp1, instances=1, episodes=100, steps=30,
                      reset_at_terminal=True, verbose=False, open_plot=True)

    run_agents_on_mdp(agent_pool, mdp2, instances=1, episodes=100, steps=30,
                      reset_at_terminal=True, verbose=False, open_plot=True)


if __name__ == "__main__":
    main()
