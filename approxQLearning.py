import gym
import random
env = gym.make('LunarLander-v2')
observation = env.reset()


# GLOBAL VARIABLES
EPSILON = 0.1
ALPHA = 0.3
DISCOUNT = 0.7

TESTING_EPISODE = 900

QVALUES = {}

# Approx Q Learning globals
WEIGHTS = {}

# HELPER FUNCTIONS

# Iterate through all of the actions from the given state, return maximum qValue from them
def computeValueFromQValues(state):
    qValues = []

    for action in range(4):
        qValues.append(getQValue(state, action))

    return max(qValues)

# Given a state, round the values inside to nearest whole number
def roundState(state):
    return tuple(map(lambda x: round(x, 0), tuple(state)))

# Given a state, update global QValues
def update(state, action, nextState, reward):

    features = extractFeatures(state, action)
    
    alphaTimesDifference = ALPHA * (reward + DISCOUNT * computeValueFromQValues(nextState) \
                                    - getQValue(state, action))

    for f in features:
        global WEIGHTS

        if f not in WEIGHTS:
            WEIGHTS[f] = 0

        # if f == 'XVelocity':
        #     print("alphaTimesDifference " + str(alphaTimesDifference))
        #     print("features[f] " + str(features[f]))
        #     print()

        WEIGHTS[f] += alphaTimesDifference * features[f]

def extractFeatures(state, action):
    features = {}

    # features['XDistFromCenter'] = abs(state[0])
    # features['YPos'] = state[1]
    features['DistanceFromCenter'] = ((state[0] ** 2) + (state[1] ** 2)) ** 0.5

    # features['XVelocity'] = state[2]
    # features['YVelocity'] = state[3]

    features['Speed'] = ((state[2] ** 2) + (state[3] ** 2)) ** 0.5

    features['Angle'] = state[4]
    features['AngleVelocity'] = state[5]
    features['LLeg'] = state[6]
    features['RLeg'] = state[7]

    return features

# Given a state and an action, return Q Value
def getQValue(state, action):
    features = extractFeatures(state, action)
    return sum(WEIGHTS[key] * features.get(key, 0) for key in WEIGHTS)

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

for episode in range(1000):
    state = env.reset()
    lastState = None
    lastAction = None

    print(WEIGHTS)
    
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


