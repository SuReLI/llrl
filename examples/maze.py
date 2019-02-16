import numpy as np
import llrl.agents.random as rd
import llrl.agents.mcts as mcts
import llrl.agents.uct as uct
from llrl.envs.gridworld import GridWorld

# Parameters
np.random.seed(1993)
timeout = 100
env = GridWorld(map_name='maze', slipperiness=0.1)

# agent = rd.RandomAgent(env.action_space)
# agent = mcts.MCTS(env.action_space)
agent = uct.UCT(env.action_space)
agent.display()

# Run
done = False
env.render()
discounted_return, total_return, total_time = 0.0, 0.0, 0
for t in range(timeout):
    action = agent.act(env, done)
    _, reward, done = env.step(action)
    total_return += reward
    discounted_return += (agent.gamma**t) * reward
    env.render()
    if (t + 1 == timeout) or done:
        total_time = t + 1
        break
print('End of episode')
print('Total time        :', total_time)
print('Total return      :', total_return)
print('Discounted return :', discounted_return)
