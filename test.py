import gym
import sys
env = gym.make('LunarLander-v2')
observation = env.reset()


# Global variables
EPSILON = 0.3
QVALUES = {}



# print(env.action_space)

for episode_index in range(1):
    observation = env.reset()

    for t in range(1000):
        env.render()
        #print(observation)

        action = env.action_space.sample()

        print(action)

        observation, reward, done, info = env.step(action)

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break


# Given a state and an action, return Q Value
def getQValue(state, action):
    if (state, action) in QVALUES.keys():
        return QVALUES[(state, action)]
    else:
        return 0.0

# Given an observation, decide an action
# Action is a discrete number between [0, 3]
def getAction(state):

    actionToQValue = {}

    # Iterate through each action
    for action in range(4):
        actionToQValue[action] = getQValue(state, action)

    optimalAction = None
    maxQValue = 


    




    # Iterate over the action space
    # Find out which promises the highest reward