import gym
import random
env = gym.make('LunarLander-v2')
observation = env.reset()


# GLOBAL VARIABLES
EPSILON = 0.1
ALPHA = 0.3
DISCOUNT = 0.7

TESTING_EPISODE = 4900

QVALUES = {}

# HELPER FUNCTIONS

# Iterate through all of the actions from the given state, return maximum qValue from them
def computeValueFromQValues(state):
    qValues = []

    for action in range(4):
        qValues.append(getQValue(state, action))

    return max(qValues)

# Given a state, update global QValues
def update(state, action, nextState, reward):
    global QVALUES
    QVALUES[(tuple(state), action)] = (1 - ALPHA) * getQValue(state, action) + (ALPHA * \
                                (reward + DISCOUNT * computeValueFromQValues(nextState)))

# Given a state and an action, return Q Value
def getQValue(state, action):
    if (tuple(state), action) in QVALUES.keys():
        return QVALUES[(state, action)]
    else:
        return 0.0

# Given an observation, decide an action based on highest QValue
# Action is a discrete number between [0, 3]
def getAction(state):

    optimalAction = None
    maxQValue = -float("inf")

    # Epsilon check for exploration (random action sampling)
    if random.random() < EPSILON:
        return env.action_space.sample()

    # Iterate through each action
    for action in range(4):
        qValue = getQValue(state, action)
        if qValue > maxQValue:
            optimalAction = action
            maxQValue = qValue

    return optimalAction

# MAIN

for episode in range(5000):
    state = env.reset()
    lastState = None
    lastAction = None
    

    for t in range(1000):
        # env.render()

        if episode > TESTING_EPISODE:
            env.render()

        if t == 0:
            # Pick random action for first step
            action = env.action_space.sample()
        else:
            action = getAction(state)

        lastState = state
        lastAction = action

        state, reward, done, info = env.step(action)
        #print(observation)

        # Update QValues based on state

        # print(action)

        update(lastState, lastAction, state, reward)

        # print(QVALUES.values())
        # print()

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

    if episode == TESTING_EPISODE:
        print('ENTERED TESTING MODE')
        EPSILON = 0
        ALPHA = 0


