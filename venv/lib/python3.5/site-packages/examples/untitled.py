# Other imports.
from simple_rl.agents import RandomAgent, LinearQAgent
from simple_rl.tasks import GymMDP
from simple_rl.run_experiments import run_agents_on_mdp

# Gym MDP
gym_mdp = GymMDP(env_name='Breakout-v0', render=True)
num_feats = gym_mdp.get_num_state_feats()

# Setup agents and run.
rand_agent = RandomAgent(gym_mdp.get_actions())
lin_q_agent = LinearQAgent(gym_mdp.get_actions(), num_feats)

# Run experiment.
run_agents_on_mdp([lin_q_agent, rand_agent], gym_mdp, instances=3, episodes=20, steps=200, verbose=True)