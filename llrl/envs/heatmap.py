import random
import numpy as np

from llrl.envs.gridworld import GridWorld
from simple_rl.tasks.grid_world.GridWorldStateClass import GridWorldState


class HeatMap(GridWorld):
    """
    Heat-map grid-world.
    """

    def __init__(
            self,
            width=5,
            height=3,
            init_loc=(1, 1),
            rand_init=False,
            goal_locs=[(5, 3)],
            lava_locs=[()],
            walls=[],
            is_goal_terminal=False,
            gamma=0.99,
            slip_prob=0.0,
            step_cost=0.0,
            lava_cost=0.01,
            goal_reward=1.0,
            reward_span=1.0,
            name="Heat-map"
    ):
        GridWorld.__init__(
            self,
            width=width,
            height=height,
            init_loc=init_loc,
            rand_init=rand_init,
            goal_locs=goal_locs,
            lava_locs=lava_locs,
            walls=walls,
            is_goal_terminal=is_goal_terminal,
            gamma=gamma,
            slip_prob=slip_prob,
            step_cost=step_cost,
            lava_cost=lava_cost,
            goal_reward=goal_reward,
            name=name
        )
        self.reward_span = reward_span

    def transition(self, s, a):
        """
        Joint transition method.

        :param s: (GridWorldState) state
        :param a: (str) action
        :return: reward and resulting state (r, s_p)
        """

        if s.is_terminal():
            return 0., s
        
        if self.slip_prob > random.random():  # Flip direction
            if a == "up":
                a = random.choice(["left", "right"])
            elif a == "down":
                a = random.choice(["left", "right"])
            elif a == "left":
                a = random.choice(["up", "down"])
            elif a == "right":
                a = random.choice(["up", "down"])

        if a == "up" and s.y < self.height and not self.is_wall(s.x, s.y + 1):
            s_p = GridWorldState(s.x, s.y + 1)
        elif a == "down" and s.y > 1 and not self.is_wall(s.x, s.y - 1):
            s_p = GridWorldState(s.x, s.y - 1)
        elif a == "right" and s.x < self.width and not self.is_wall(s.x + 1, s.y):
            s_p = GridWorldState(s.x + 1, s.y)
        elif a == "left" and s.x > 1 and not self.is_wall(s.x - 1, s.y):
            s_p = GridWorldState(s.x - 1, s.y)
        else:
            s_p = GridWorldState(s.x, s.y)

        if (s_p.x, s_p.y) in self.goal_locs and self.is_goal_terminal:
            s_p.set_terminal(True)

        if (s_p.x, s_p.y) in self.goal_locs:
            r = self.goal_reward - self.step_cost
        elif (s_p.x, s_p.y) in self.lava_locs:
            r = - self.lava_cost
        else:
            heat_reward = 0.
            for g in self.goal_locs:
                heat_reward += np.exp(- ((s_p.x - g[0])**2 + (s_p.y - g[1])**2) / (2. * self.reward_span**2))
            r = heat_reward - self.step_cost

        return r, s_p

    def _reward_func(self, state, action):
        raise ValueError('Method _reward_func not implemented in this Grid-world version, see transition method.')

    def _transition_func(self, state, action):
        raise ValueError('Method _transition_func not implemented in this Grid-world version, see transition method.')

    def states(self):
        """
        Compute a list of the states of the environment.
        :return: list of states
        """
        states = []
        for i in range(1, self.width + 1):
            for j in range(1, self.height + 1):
                s = GridWorldState(i, j)
                if (i, j) in self.goal_locs and self.is_goal_terminal:
                    s.set_terminal(True)
                states.append(s)
        return states

