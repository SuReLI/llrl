"""
UCT Algorithm

Required features of the environment:
env.state
env.action_space
env.transition(s, a)
env.equality_operator(s1, s2)
"""

import llrl.agents.mcts as mcts
import llrl.utils.utils as utils
import llrl.spaces.discrete as discrete
from math import sqrt, log


def uct_tree_policy(ag, children):
    return max(children, key=ag.ucb)


class UCT(object):
    """
    UCT agent
    """
    def __init__(self, action_space, rollouts=100, horizon=100, gamma=0.9, ucb_constant=6.36396103068):
        self.action_space = action_space
        self.n_actions = self.action_space.n
        self.rollouts = rollouts
        self.horizon = horizon
        self.gamma = gamma
        self.ucb_constant = ucb_constant

    def reset(self, p=None):
        """
        Reset the attributes.
        Expect to receive them in the same order as init.
        p : list of parameters
        """
        if p is None:
            self.__init__(self.action_space)
        else:
            utils.assert_types(p, [discrete.Discrete, int, int, float, float])
            self.__init__(p[0], p[1], p[2], p[3], p[4])

    def display(self):
        """
        Display info about the attributes.
        """
        print('Displaying UCT agent:')
        print('Action space       :', self.action_space)
        print('Number of actions  :', self.n_actions)
        print('Rollouts           :', self.rollouts)
        print('Horizon            :', self.horizon)
        print('Gamma              :', self.gamma)
        print('UCB constant       :', self.ucb_constant)

    def ucb(self, node):
        """
        Upper Confidence Bound of a chance node
        """
        return mcts.chance_node_value(node) + self.ucb_constant * sqrt(log(node.parent.visits)/len(node.sampled_returns))

    def act(self, env, done):
        return mcts.mcts_procedure(self, uct_tree_policy, env, done)
