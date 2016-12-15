import gym
from gym import envs

'''
env = gym.make('CartPole-v0')
#env.monitor.start('/Users/neural1977/Desktop/cartpole-experiment1')
for i_episode in range(20):
    observation = env.reset();
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

#env.monitor.close()
'''
#env = gym.make('MountainCar-v0')

env = gym.make('CartPole-v0')
env.reset()
print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)
print (envs.registry.all())
for _ in range(1000):
    env.render()
    r = env.step(env.action_space.sample())     # take a random action
    #print r
