import numpy as np

from simple_rl.agents import QLearningAgent, RandomAgent, RMaxAgent
from simple_rl.run_experiments import run_agents_on_mdp
from llrl.agents.rmax import RMax
from llrl.envs.gridworld import GridWorld


def main():
    # Setup MDP.
    w = 6
    h = 6
    mdp = GridWorld(width=w, height=h, init_loc=(1, 1), goal_locs=[(6, 6)], slip_prob=.1)

    # Setup Agents.
    rand_agent = RandomAgent(actions=mdp.get_actions())
    ql_agent = QLearningAgent(actions=mdp.get_actions())

    # Compute number of samples for R-MAX to achieve epsilon optimal behavior with high probability (1 - delta)
    compute_n_samples = False
    if compute_n_samples:
        epsilon = .1
        delta = .05
        m_r = np.log(2. / delta) / (2. * epsilon ** 2)
        m_t = 2. * (np.log(2**(float(w * h)) - 2.) - np.log(delta)) / (epsilon ** 2)
        n_samples = int(max(m_r, m_t))
    else:
        n_samples = 30

    simple_rl_rmax_agent = RMaxAgent(
        actions=mdp.get_actions(), gamma=.9, horizon=3, s_a_threshold=n_samples, name='SimpleRL-R-MAX'
    )
    rmax_agent = RMax(actions=mdp.get_actions(), gamma=.9, count_threshold=n_samples)

    # Run experiment and make plot.
    run_agents_on_mdp(
        [rand_agent, ql_agent, rmax_agent, simple_rl_rmax_agent], mdp, instances=5,
        episodes=100, steps=20, reset_at_terminal=True, verbose=False
    )


if __name__ == "__main__":
    main()
