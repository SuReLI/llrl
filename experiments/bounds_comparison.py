"""
Comparison of the bounds provided by R-MAX and Lipschitz-R-Max.
"""

from llrl.envs.twostates import TwoStates
from llrl.agents.lrmax import LRMax
from llrl.agents.rmax_vi import RMaxVI
from simple_rl.agents import RandomAgent
from simple_rl.run_experiments import run_agents_on_mdp
from simple_rl.tasks import GridWorldMDP


def test(agent):
    actions = agent.actions
    print('test:')

    print('len U_mem:', len(agent.U_memory), 'len U_mem[0]', len(agent.U_memory[0]))
    print('len R_mem:', len(agent.R_memory), 'len R_mem[0]', len(agent.R_memory[0]))
    print('len T_mem:', len(agent.T_memory), 'len T_mem[0]', len(agent.T_memory[0]))
    for s in agent.U_memory[0]:
        for a in actions:
            print(s, a, 'U(s,a) =', agent.U_memory[0][s][a])

def main():
    # MDP
    mdp1 = GridWorldMDP(width=5, height=5, init_loc=(1, 1), goal_locs=[(5, 5)])
    # mdp1 = TwoStates(rewards=(0., 1.), proba=(0., 0.))
    # mdp2 = TwoStates(rewards=(0., 1.), proba=(.9, .9))

    # Agents
    rand_agent = RandomAgent(actions=mdp1.get_actions())
    rmax_vi_agent = RMaxVI(actions=mdp1.get_actions(), gamma=.9, count_threshold=1)
    lrmax_agent = LRMax(actions=mdp1.get_actions(), gamma=0.9, count_threshold=1)

    # Run
    run_agents_on_mdp([lrmax_agent, rmax_vi_agent, rand_agent], mdp1, instances=1, episodes=30, steps=20,
                      reset_at_terminal=True, verbose=False, open_plot=True)
    test(lrmax_agent)


if __name__ == "__main__":
    main()
