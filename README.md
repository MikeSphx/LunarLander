# OpenAI Gym Lunar Lander Project
### Michael Kim, Seung Seok Lee

This is the codebase for our CS4100 project where we attempted to use reinforcement learning to solve the OpenAI Gym Lunar Lander problem.

## Setup
The following is required to run the environment:
- Python 3
- OpenAI (pip install gym)
- Numpy (pip install numpy)
- Matplotlib (pip install matplotlib)

Once setup, simply run the following command on any of the files listed in the Directory:

`python <file>`

## Directory
Listed in the order in which we implemented them:
- roundingQLearning.py : Our "rounded" Q-Learning approach
- approxQLearning.py : First attempt at using approximate Q-Learning to reduce state space
- modelFreeQLearning.py : Correction to approxQLearning where we added action-based features + E-decay
- supervisedLearning.py : Implementing the heuristic to help with supervised learning
- analyticsSingleGraph.py : Same as supervisedLearning but with Matplotlib code for creating dot+line plots for paper
- analyticsMultiGraph.py : Same as supervisedLearning but with Matplotlib code for creating multiple-line plots for paper
