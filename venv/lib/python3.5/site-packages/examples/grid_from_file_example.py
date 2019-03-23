#!/usr/bin/env python

# Python imports.
import sys

# Other imports.
import srl_example_setup
from simple_rl.agents import QLearningAgent, RandomAgent, RMaxAgent
from simple_rl.tasks import GridWorldMDP
from simple_rl.tasks.grid_world.GridWorldMDPClass import make_grid_world_from_file
from simple_rl.run_experiments import run_agents_on_mdp

def main(open_plot=True):
    # Setup MDP, Agents.
    mdp = make_grid_world_from_file("octo_grid.txt")

    ql_agent = QLearningAgent(actions=mdp.get_actions())
    rand_agent = RandomAgent(actions=mdp.get_actions())

    visualize = True
    if visualize:
    	mdp.visualize_learning(ql_agent)
    else:
	    # Run experiment and make plot.
	    run_agents_on_mdp([ql_expl_agent, ql_min_agent, rand_agent], mdp, instances=25, episodes=50, steps=25, open_plot=open_plot, track_disc_reward=False)

if __name__ == "__main__":
    main(open_plot=not sys.argv[-1] == "no_plot")
