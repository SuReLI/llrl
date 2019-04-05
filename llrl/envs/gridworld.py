from simple_rl.tasks import GridWorldMDP
from simple_rl.tasks.grid_world.GridWorldStateClass import GridWorldState

class GridWorld(GridWorldMDP):
    """
    Copy of GridWorldMDP from simple_rl with a few changes:
    - Reward when reaching goal is set as an attribute
    """

    def __init__(self,
                 width=5,
                 height=3,
                 init_loc=(1, 1),
                 rand_init=False,
                 goal_locs=[(5, 3)],
                 lava_locs=[()],
                 walls=[],
                 is_goal_terminal=True,
                 gamma=0.99,
                 slip_prob=0.0,
                 step_cost=0.0,
                 lava_cost=0.01,
                 goal_reward=1.0,
                 name="gridworld"):
        GridWorldMDP.__init__(self,
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
                              name=name)
        self.goal_reward = goal_reward

    def _reward_func(self, state, action):
        """
        Override of the reward function of GridWorldMDP setting a variable goal reward
        when reaching the goal (set in the attribute self.goal_reward)
        :param state: queried state
        :param action: queried action
        :return: None
        """
        if self._is_goal_state_action(state, action):
            return self.goal_reward - self.step_cost
        elif (int(state.x), int(state.y)) in self.lava_locs:
            return -self.lava_cost
        else:
            return 0 - self.step_cost

    def reward_func(self, state, action):
        return self._reward_func(state, action)

    def transition_func(self, state, action):
        return self._transition_func(state, action)

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

