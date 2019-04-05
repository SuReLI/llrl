import numpy as np

from simple_rl.agents import QLearningAgent, RandomAgent, RMaxAgent
from simple_rl.tasks import GridWorldMDP
from simple_rl.run_experiments import run_agents_on_mdp
from llrl.agents.fast_rmax_vi import RMaxVI as FastRMaxVI
from llrl.agents.rmax_vi import RMaxVI


def main():
    # Setup MDP.
    w = 6
    h = 6
    mdp = GridWorldMDP(width=w, height=h, init_loc=(1, 1), goal_locs=[(6, 6)], slip_prob=.1)

    # Compute number of samples for R-MAX to achieve epsilon optimal behavior with high probability (1 - delta)
    epsilon = .1
    delta = .05
    m_r = np.log(2. / delta) / (2. * epsilon ** 2)
    m_t = 2. * (np.log(2**(float(w * h)) - 2.) - np.log(delta)) / (epsilon ** 2)
    n_samples = int(max(m_r, m_t))

    # Setup Agents.
    ql_agent = QLearningAgent(actions=mdp.get_actions())
    rand_agent = RandomAgent(actions=mdp.get_actions())
    rmax_agent = RMaxAgent(actions=mdp.get_actions(), gamma=.9, horizon=3, s_a_threshold=n_samples)

    fast_rmax_vi_agent = FastRMaxVI(actions=mdp.get_actions(), gamma=.9, count_threshold=n_samples, name='fast-rmax-vi')
    rmax_vi_agent = RMaxVI(actions=mdp.get_actions(), gamma=.9, count_threshold=n_samples)

    # Run experiment and make plot.
    run_agents_on_mdp(
        [fast_rmax_vi_agent, rmax_vi_agent, rmax_agent, ql_agent, rand_agent], mdp, instances=5,
        episodes=100, steps=20, reset_at_terminal=True, verbose=False
    )


if __name__ == "__main__":
    main()
