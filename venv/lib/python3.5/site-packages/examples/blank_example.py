from simple_rl.tasks import GridWorldMDP, CartPoleMDP
from simple_rl.agents import QLearningAgent, RandomAgent
from simple_rl.run_experiments import run_agents_on_mdp

# Setup MDP.
# mdp = GridWorldMDP(width=4, height=3, init_loc=(1, 1), goal_locs=[(4, 3)])
mdp = CartPoleMDP()

# Make agents.
ql_agent = QLearningAgent(actions=mdp.get_actions())
rand_agent = RandomAgent(actions=mdp.get_actions())

# Run experiment and make plot.
run_agents_on_mdp([rand_agent, ql_agent], mdp, instances=10, episodes=50, steps=100)


def reproduce(open_plot=True):
    from simple_rl.run_experiments import reproduce_from_exp_file
    import os

    # Grab results directory.
    results_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "results")

    # Reproduce the experiment.
    reproduce_from_exp_file(exp_name="gridworld_h-3_w-4", results_dir=results_dir, open_plot=open_plot)

# if __name__ == "__main__":
#     main()
    # reproduce()
