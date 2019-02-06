import numpy as np
import llrl.envs.gridworld as gridworld
import llrl.agents.random as rd

# Parameters
np.random.seed(1993)
timeout = 100
env = gridworld.Gridworld(map_name='maze', nT=timeout)
agent = rd.RandomAgent(env.action_space)
agent.display()

# Run
done = False
env.render()
discounted_return, total_return, total_time = 0.0, 0.0, 0
for t in range(timeout):
    action = agent.act(env, done)
    _, reward, done, _ = env.step(action)
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
