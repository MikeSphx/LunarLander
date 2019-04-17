import gym
import random
import numpy as np
env = gym.make('LunarLander-v2')
observation = env.reset()


# GLOBAL VARIABLES
# EPSILON = 0.995

EPSILON = 0.9995
EDECAY = 0.99941
ALPHA = 0.001
DISCOUNT = 0.9995
SUPERVISED_EPISODE = 2

EXPERIMENTS = {}
EXPERIMENTS['2 Supervised'] = ({'EP': 0.9995, 'ED':0.99941, 'AL':0.001, 'DI':0.9995, 'SU':2})

TESTING_EPISODE = 5000
TOTAL_EPISODES = 5500
EPISODE_TIME_LIMIT = 1000

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

    features['XPos0'] = state[0] if action == 0 else 0
    features['XPos1'] = state[0] if action == 1 else 0
    features['XPos2'] = state[0] if action == 2 else 0
    features['XPos3'] = state[0] if action == 3 else 0

    features['YPos0'] = state[1] if action == 0 else 0
    features['YPos1'] = state[1] if action == 1 else 0
    features['YPos2'] = state[1] if action == 2 else 0
    features['YPos3'] = state[1] if action == 3 else 0

    # CUSTOM
    #features['DistanceFromCenter'] = ((state[0] ** 2) + (state[1] ** 2)) ** 0.5

    # distFromCenter = ((state[0] ** 2) + (state[1] ** 2)) ** 0.5
    # features['DistanceFromCenter0'] = distFromCenter if action == 0 else 0
    # features['DistanceFromCenter1'] = distFromCenter if action == 1 else 0
    # features['DistanceFromCenter2'] = distFromCenter if action == 2 else 0
    # features['DistanceFromCenter3'] = distFromCenter if action == 3 else 0

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

def heuristic(env, s):
    # Heuristic for:
    # 1. Testing. 
    # 2. Demonstration rollout.
    angle_targ = s[0]*0.5 + s[2]*1.0         # angle should point towards center (s[0] is horizontal coordinate, s[2] hor speed)
    if angle_targ >  0.4: angle_targ =  0.4  # more than 0.4 radians (22 degrees) is bad
    if angle_targ < -0.4: angle_targ = -0.4
    hover_targ = 0.55*np.abs(s[0])           # target y should be proporional to horizontal offset

    # PID controller: s[4] angle, s[5] angularSpeed
    angle_todo = (angle_targ - s[4])*0.5 - (s[5])*1.0
    #print("angle_targ=%0.2f, angle_todo=%0.2f" % (angle_targ, angle_todo))

    # PID controller: s[1] vertical coordinate s[3] vertical speed
    hover_todo = (hover_targ - s[1])*0.5 - (s[3])*0.5
    #print("hover_targ=%0.2f, hover_todo=%0.2f" % (hover_targ, hover_todo))

    if s[6] or s[7]: # legs have contact
        angle_todo = 0
        hover_todo = -(s[3])*0.5  # override to reduce fall speed, that's all we need after contact

    a = 0
    if hover_todo > np.abs(angle_todo) and hover_todo > 0.05: a = 2
    elif angle_todo < -0.05: a = 3
    elif angle_todo > +0.05: a = 1

    # if env.continuous:
    #     a = np.array( [hover_todo*20 - 1, -angle_todo*20] )
    #     a = np.clip(a, -1, +1)
    # else:
    #     a = 0
    #     if hover_todo > np.abs(angle_todo) and hover_todo > 0.05: a = 2
    #     elif angle_todo < -0.05: a = 3
    #     elif angle_todo > +0.05: a = 1

    return a

# MAIN
# Returns a Pandas DataFrame
def main():

    experiment_results = []
    experiment_results.append()

    average_episode_frame = 500
    cumulative_score = 0

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

            # if episode > TESTING_EPISODE or episode % SUPERVISED_EPISODE == 0: # or episode % 100 == 0:
                # env.render()

            if episode > TESTING_EPISODE:
                env.render()

            if episode % SUPERVISED_EPISODE == 0 and episode < TESTING_EPISODE:
                action = heuristic(env, state)
            else:
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

            if done:
                # print("Episode {} finished after {} timesteps. Score: {}".format(episode, t+1, currentScore))
                cumulative_score += currentScore
                if episode % average_episode_frame == 0 and episoe <= TESTING_EPISODE:
                    experiment_results.append(cumulative_score / average_episode_frame)
                break

        if episode == TESTING_EPISODE:
            print('\a')
            print('\a')
            print('\a')
            print('\a')
            print('\a')
            print('ENTERED TESTING MODE')
            print(WEIGHTS)
            EPSILON = 0
            ALPHA = 0

    return experiment_results

data = {}

for key, e in EXPERIMENTS:
    global EPSILON, EDECAY, ALPHA, DISCOUNT, SUPERVISED_EPISODE, QVALUES, WEIGHTS
    EPSILON = e.EP
    EDECAY = e.ED
    ALPHA = e.AL
    DISCOUNT = e.DI
    SUPERVISED_EPISODE = e.SU
    QVALUES = {}
    WEIGHTS = {}

    data[key] = main()

columns = ['Parameters', '500', '1000', '1500', '2000', '2500', '3000', '3500', '4000', '4500', '5000', 'Weights']


