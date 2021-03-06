import gym
import random
env = gym.make('LunarLander-v2')
observation = env.reset()


# GLOBAL VARIABLES
# EPSILON = 0.99
# ALPHA = 0.001
# DISCOUNT = 0.99
EPSILON = 0.1
ALPHA = 0.3
DISCOUNT = 0.7

TESTING_EPISODE = 4900
TOTAL_EPISODES = 5000
EPISODE_TIME_LIMIT = 1000
TESTING_SCORES = []

QVALUES = {}

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
    global QVALUES
    roundedStateTuple = roundState(state)
    QVALUES[(roundedStateTuple, action)] = (1 - ALPHA) * getQValue(state, action) + (ALPHA * \
                                (reward + DISCOUNT * computeValueFromQValues(nextState)))

# Given a state and an action, return Q Value
def getQValue(state, action):
    roundedStateTuple = roundState(state)
    if (roundedStateTuple, action) in QVALUES.keys():
        return QVALUES[(roundedStateTuple, action)]
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

for episode in range(TOTAL_EPISODES):
    state = env.reset()
    lastState = None
    lastAction = None

    currentScore = 0
    
    for t in range(EPISODE_TIME_LIMIT):
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

        currentScore += reward

        update(lastState, lastAction, state, reward)

        # print(QVALUES.values())
        # print()

        if done:
            # print("Episode finished after {} timesteps".format(t+1))
            print("Episode {} finished after {} timesteps. Score: {}".format(episode, t+1, currentScore))
            break

    if episode == TESTING_EPISODE:
        print('ENTERED TESTING MODE')
        EPSILON = 0
        ALPHA = 0
    elif episode > TESTING_EPISODE:
        print('Testing episode finished with score: ' + str(currentScore))
        TESTING_SCORES.append(currentScore)

print('Average score over testing episodes: ' + str(mean(TESTING_SCORES)))
