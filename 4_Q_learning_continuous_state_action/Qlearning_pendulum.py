'''
The inverted pendulum swingup problem is a classic problem in the control literature. 
In this version of the problem, the pendulum starts in a random position
and the goal is to swing it up so it stays upright.

In this example, the state and action spaces are continuous.
To use Q-Learning, it is necessary to discretize the continuous 
state and action spaces into a number of buckets.
'''

import gym
import numpy as np
import math
import matplotlib.pyplot as plt
from gym import wrappers
from time import time 


# INITIALISE ENVIRONMENT
env = gym.make('Pendulum-v0')
n_actions = env.action_space.shape[0]
n_states = env.observation_space.shape[0]
print("Action space size: ", n_actions)
print("State space size: ", n_states)

print('Action boundaries:', env.action_space.high, env.action_space.low)
print('State boundaries:', env.observation_space.high, env.observation_space.low)

# The action is the joint effort between -2.0 and 2.0
# The state is a vector of the following values:
# cos(theta)	-1.0	1.0
# sin(theta)	-1.0	1.0
# theta dot	-8.0	8.0

# HYPERPARAMETERS
n_episodes = 1000           # Total train episodes
n_steps = 200               # Max steps per episode
min_alpha = 0.06             # learning rate | best: 0.06
min_epsilon = 0.01           # exploration rate | best: 0.01
gamma = 0.91                # discount factor | best: 0.91
decay_epsilon = 3          # decay rate parameter for epsilon | best: 3
decay_alpha = 23            # decay rate parameter for alpha | best: 23

action_buckets = 2     # best: 2
state_buckets = (13, 11, 12)  # best: (13, 11, 12)

ub_action = env.action_space.high
lb_action = env.action_space.low

ub_state = env.observation_space.high
lb_state = env.observation_space.low


def discretize(obs, lower_bounds, upper_bounds, buckets):
    ''' discretise the continuous state into buckets ''' 
    ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
    new_obs = [int(round((buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
    new_obs = [min(buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
    return tuple(new_obs)

def discretize_action(action, lower_bounds, upper_bounds, buckets):
    ''' discretise the continuous action into buckets ''' 
    ratios = (action + abs(lower_bounds)) / (upper_bounds - lower_bounds)
    new_action = int(np.round((buckets - 1) * ratios))
    new_action = min(buckets - 1, max(0, new_action))
    res = (new_action,)    # need to convert int to tuple
    return res

def convert_action_to_float(x):
    """ 
    x is a int between 0 and 4. The function scale x between -2 and 2. 
    """
    OldMax = action_buckets-1 
    OldMin = 0
    NewMax = 2
    NewMin = -2

    OldRange = (OldMax - OldMin)  
    NewRange = (NewMax - NewMin)  
    y = (((x - OldMin) * NewRange) / OldRange) + NewMin
    res = [y] # need to convert to a list to be readable by the step function
    return res

def epsilon_policy(state, epsilon):
    ''' choose an action using the epsilon policy '''
    exploration_exploitation_tradeoff = np.random.random()
    if exploration_exploitation_tradeoff <= epsilon:
        action = env.action_space.sample()  # exploration
    else:
        discrete_action = np.argmax(Q[state])   # exploitation
        action = convert_action_to_float(discrete_action)
    return action

def greedy_policy(state):
    ''' choose an action using the greedy policy '''
    discrete_action = np.argmax(Q[state])  
    action = convert_action_to_float(discrete_action)
    return action

def update_q(current_state, action, reward, new_state, alpha):
    ''' update the Q matrix with the Bellman equation '''
    current_state_action_pair = current_state + action
    Q[current_state_action_pair] += alpha * (reward + gamma * np.max(Q[new_state]) - Q[current_state_action_pair])

def get_epsilon(t):
    ''' decrease the exploration rate at each episode '''
    return max(min_epsilon, min(1, 1.0 - math.log10((t + 1) / decay_epsilon)))

def get_alpha(t):
    ''' decrease the learning rate at each episode '''
    return max(min_alpha, min(1.0, 1.0 - math.log10((t + 1) / decay_alpha)))


# INITIALISE Q MATRIX
Q = np.zeros(state_buckets + (action_buckets,)) 
print(Q.shape)

# TRAINING PHASE
rewards = [] 

for episode in range(n_episodes):
    current_state = env.reset()
    current_state = discretize(current_state, lb_state, ub_state, state_buckets)

    alpha = get_alpha(episode)
    epsilon = get_epsilon(episode)

    episode_rewards = 0

    for t in range(n_steps):
        action = epsilon_policy(current_state, epsilon)
        new_state, reward, done, _ = env.step(action)
        new_state = discretize(new_state, lb_state, ub_state, state_buckets)
        action = discretize_action(action, lb_action, ub_action, action_buckets)
        update_q(current_state, action, reward, new_state, alpha)
        current_state = new_state
        episode_rewards += reward

        if done:
            print('episode: {0:d}/{1:d} | Reward: {2:08.02f} | Epsilon: {3:.2f} | Alpha: {4:.2f}'.format(episode, n_episodes, episode_rewards, epsilon, alpha))
            break

    # append the episode cumulative reward to the reward list
    rewards.append(episode_rewards)


# PLOT RESULTS
x = range(n_episodes)
plt.plot(x, rewards)
plt.xlabel('Episodes')
plt.ylabel('Training cumulative reward')
plt.savefig('plots/Qlearning_pendulum.png', dpi=300)
plt.show()

# TEST PHASE
# record video
env = wrappers.Monitor(env, './plots/' + str(time()) + '/')

for e in range(10):

    current_state = env.reset()
    current_state = discretize(current_state, lb_state, ub_state, state_buckets)
    episode_rewards = 0

    for t in range(n_steps):
        env.render()
        action = greedy_policy(current_state)
        new_state, reward, done, _ = env.step(action)
        new_state = discretize(new_state, lb_state, ub_state, state_buckets)
        action = discretize_action(action, lb_action, ub_action, action_buckets)
        update_q(current_state, action, reward, new_state, alpha)
        current_state = new_state
        episode_rewards += reward

        if done:
            print('Test episode finished with a total reward of: {:.2f}'.format(episode_rewards))
            break

env.close()

