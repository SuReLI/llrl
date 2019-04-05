from simple_rl.agents import RandomAgent
from simple_rl.run_experiments import run_agents_on_mdp
from llrl.agents.rmax import RMax
from llrl.agents.lrmax import LRMax
from llrl.envs.gridworld import GridWorld


def example():
    env = GridWorld(width=6, height=6, init_loc=(1, 1), goal_locs=[(6, 6)], slip_prob=.1)

    rand = RandomAgent(actions=env.get_actions())
    rmax = RMax(actions=env.get_actions(), gamma=.9, count_threshold=1)
    lrmax = LRMax(actions=env.get_actions(), gamma=.9, count_threshold=1)

    run_agents_on_mdp(
        [rand, rmax, lrmax], env, instances=5, episodes=100, steps=20, reset_at_terminal=True, verbose=False
    )


def test():
    print('test')


if __name__ == "__main__":
    example()
