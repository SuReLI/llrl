from simple_rl.agents import QLearningAgent, RandomAgent, RMaxAgent
from simple_rl.run_experiments import run_agents_on_mdp
from llrl.envs.twostates import TwoStates
from llrl.agents.rmax import RMax  # TODO remove

def main():
    # Setup MDP.
    mdp = TwoStates()

    # Setup Agents.
    ql_agent = QLearningAgent(actions=mdp.get_actions())
    rand_agent = RandomAgent(actions=mdp.get_actions())
    rmax_agent = RMaxAgent(actions=mdp.get_actions(), gamma=.9, horizon=3, s_a_threshold=10)
    my_rmax = RMax(actions=mdp.get_actions(), gamma=.9, horizon=3, count_threshold=1, name="my-rmax")  # TODO remove

    # Run experiment and make plot.
    # run_agents_on_mdp([rmax_agent, ql_agent, rand_agent], mdp, instances=5,  # TODO put back
    run_agents_on_mdp([rmax_agent, my_rmax, ql_agent, rand_agent], mdp, instances=5,
                      episodes=100, steps=20, reset_at_terminal=True, verbose=False)

if __name__ == "__main__":
    main()
