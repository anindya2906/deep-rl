import gym


env = gym.make('CartPole-v0')

state = env.reset()

for i in range(1000):
    env.render()
    action = env.action_space.sample()
    next_state, reward, info, done = env.step(action)
    if done:
        _ = env.reset()
        break
