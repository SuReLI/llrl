from simple_rl.agents import QLearningAgent, RandomAgent, RMaxAgent
from simple_rl.tasks import GridWorldMDP
from simple_rl.run_experiments import run_agents_on_mdp

from llrl.agents.fast_rmax_vi import RMaxVI as FastRMaxVI
from llrl.agents.rmax_vi import RMaxVI


def main():
    # Setup MDP.
    mdp = GridWorldMDP(width=6, height=6, init_loc=(1, 1), goal_locs=[(6, 6)], slip_prob=.1)

    # Setup Agents.
    ql_agent = QLearningAgent(actions=mdp.get_actions())
    rand_agent = RandomAgent(actions=mdp.get_actions())
    rmax_agent = RMaxAgent(actions=mdp.get_actions(), gamma=.9, horizon=3, s_a_threshold=10)

    fast_rmax_vi_agent = FastRMaxVI(actions=mdp.get_actions(), gamma=.9, count_threshold=10, name='fast-rmax-vi')
    rmax_vi_agent = RMaxVI(actions=mdp.get_actions(), gamma=.9, count_threshold=10)

    # Run experiment and make plot.
    run_agents_on_mdp(
        [fast_rmax_vi_agent, rmax_vi_agent, rmax_agent, ql_agent, rand_agent], mdp, instances=5,
        episodes=100, steps=20, reset_at_terminal=True, verbose=False
    )


if __name__ == "__main__":
    main()
