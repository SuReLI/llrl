"""
Comparison of the bounds provided by R-MAX and Lipschitz-R-Max.
"""

from llrl.envs.twostates import TwoStates
from llrl.agents.lrmax import LRMax
from simple_rl.run_experiments import run_agents_on_mdp


def main():
    mdp1 = TwoStates(rewards=(0., 1.), proba=(.5, .5))
    # mdp2 = TwoStates(rewards=(0., 1.), proba=(.9, .9))

    lrmax = LRMax(actions=mdp1.get_actions(), gamma=0.9, horizon=3, count_threshold=10)

    run_agents_on_mdp([lrmax], mdp1, instances=5, episodes=100, steps=20,
                      reset_at_terminal=True, verbose=False)


if __name__ == "__main__":
    main()
