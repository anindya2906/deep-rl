import gym
from agent import Agent


env = gym.make('LunarLander-v2')

print(env.observation_space)
print(env.action_space)

# agent = Agent(env.observation_space.n, env.action_space.n, 0, 0.8)
# state = env.reset()

# while True:
#     env.render()
#     action = agent.act(state)
#     next_state, rewards, done, info = env.step(action)
#     if done:
#         break
#     state = next_state
