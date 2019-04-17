import gym
import random
env = gym.make('LunarLander-v2')
observation = env.reset()


# GLOBAL VARIABLES
# EPSILON = 0.995

# EPSILON = 0.9995
# EDECAY = 0.99941
# ALPHA = 0.001
# DISCOUNT = 0.9995

EDECAY = 1
EPSILON = 0.1
ALPHA = 0.001
DISCOUNT = 0.8

TESTING_EPISODE = 5000
TOTAL_EPISODES = 5100
EPISODE_TIME_LIMIT = 5000

QVALUES = {}

# Approx Q Learning globals
WEIGHTS = {}

# HELPER FUNCTIONS

# Iterate through all of the actions from the given state, return maximum qValue from them
def computeValueFromQValues(state):
    qValues = []

    for action in range(4):
        qValues.append(getQValue(state, action))

    bad = max(qValues)
    # print(bad)

    return bad

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
    # print(WEIGHTS)

def extractFeatures(state, action):
    features = {}

    # features['XPos'] = state[0]
    # features['YPos'] = state[1]

    # features['XPos0'] = state[0] if action == 0 else 0
    # features['XPos1'] = state[0] if action == 1 else 0
    # features['XPos2'] = state[0] if action == 2 else 0
    # features['XPos3'] = state[0] if action == 3 else 0

    # features['YPos0'] = state[1] if action == 0 else 0
    # features['YPos1'] = state[1] if action == 1 else 0
    # features['YPos2'] = state[1] if action == 2 else 0
    # features['YPos3'] = state[1] if action == 3 else 0

    # CUSTOM
    #features['DistanceFromCenter'] = ((state[0] ** 2) + (state[1] ** 2)) ** 0.5

    distFromCenter = ((state[0] ** 2) + (state[1] ** 2)) ** 0.5
    features['DistanceFromCenter0'] = distFromCenter if action == 0 else 0
    features['DistanceFromCenter1'] = distFromCenter if action == 1 else 0
    features['DistanceFromCenter2'] = distFromCenter if action == 2 else 0
    features['DistanceFromCenter3'] = distFromCenter if action == 3 else 0

    # features['XVelocity'] = state[2]
    # features['YVelocity'] = state[3]

    features['XVelocity0'] = state[2] if action == 0 else 0
    features['XVelocity1'] = state[2] if action == 1 else 0
    features['XVelocity2'] = state[2] if action == 2 else 0
    features['XVelocity3'] = state[2] if action == 3 else 0

    features['YVelocity0'] = state[3] if action == 0 else 0
    features['YVelocity1'] = state[3] if action == 1 else 0
    features['YVelocity2'] = state[3] if action == 2 else 0
    features['YVelocity3'] = state[3] if action == 3 else 0

    # CUSTOM
    # features['Speed'] = ((state[2] ** 2) + (state[3] ** 2)) ** 0.5

    # speed = ((state[2] ** 2) + (state[3] ** 2)) ** 0.5
    # features['Speed0'] = speed if action == 0 else 0
    # features['Speed1'] = speed if action == 1 else 0
    # features['Speed2'] = speed if action == 2 else 0
    # features['Speed3'] = speed if action == 3 else 0

    # features['Angle'] = state[4]
    # features['AngleVelocity'] = state[5]
    features['LLeg'] = state[6]
    features['RLeg'] = state[7]

    features['Angle0'] = state[4] if action == 0 else 0
    features['Angle1'] = state[4] if action == 1 else 0
    features['Angle2'] = state[4] if action == 2 else 0
    features['Angle3'] = state[4] if action == 3 else 0

    features['AngleVelocity0'] = state[5] if action == 0 else 0
    features['AngleVelocity1'] = state[5] if action == 1 else 0
    features['AngleVelocity2'] = state[5] if action == 2 else 0
    features['AngleVelocity3'] = state[5] if action == 3 else 0

    # features['LLeg0'] = state[6] if action == 0 else 0
    # features['LLeg1'] = state[6] if action == 1 else 0
    # features['LLeg2'] = state[6] if action == 2 else 0
    # features['LLeg3'] = state[6] if action == 3 else 0

    # features['RLeg0'] = state[7] if action == 0 else 0
    # features['RLeg1'] = state[7] if action == 1 else 0
    # features['RLeg2'] = state[7] if action == 2 else 0
    # features['RLeg3'] = state[7] if action == 3 else 0

    # Model-free action feature
    # These do not work
    # features['Action'] = action
    # features['Action0'] = int(action == 0)
    # features['Action1'] = int(action == 1)
    # features['Action2'] = int(action == 2)
    # features['Action3'] = int(action == 3)

    return features

# Given a state and an action, return Q Value
def getQValue(state, action):
    features = extractFeatures(state, action)


    bad = sum(WEIGHTS[key] * features.get(key, 0) for key in WEIGHTS)
    # print(bad)

    return bad

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

for episode in range(TOTAL_EPISODES):
    state = env.reset()
    lastState = None
    lastAction = None

    EPSILON = EPSILON * EDECAY

    # if episode > TESTING_EPISODE or episode % 100 == 0:
        # print(WEIGHTS)

    currentScore = 0
    
    for t in range(EPISODE_TIME_LIMIT):
        # env.render()

        if episode > TESTING_EPISODE: # or episode % 100 == 0:
            env.render()

        if t == 0:
            # Pick random action for first step
            action = env.action_space.sample()
        else:
            action = getAction(state)

        lastState = state
        lastAction = action

        state, reward, done, info = env.step(action)

        # print(reward)

        currentScore += reward

        update(lastState, lastAction, state, reward)

        # print(QVALUES.values())
        # print()

        if done:
            print("Episode {} finished after {} timesteps. Score: {}".format(episode, t+1, currentScore))
            break

    if episode == TESTING_EPISODE:
        print('ENTERED TESTING MODE')
        EPSILON = 0
        ALPHA = 0


