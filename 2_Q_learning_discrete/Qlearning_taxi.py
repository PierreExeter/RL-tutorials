import numpy as np
import matplotlib.pyplot as plt
import gym
import random


# INITIALISE THE ENVIRONMENT
env = gym.make("Taxi-v3")
action_size = env.action_space.n
state_size = env.observation_space.n
print("Action space size: ", action_size)
print("State space size: ", state_size)

# INITIALISE Q TABLE TO ZERO
Q = np.zeros((state_size, action_size))

# HYPERPARAMETERS
train_episodes = 2000         # Total train episodes
test_episodes = 10            # Total test episodes
n_steps = 100                 # Max steps per episode
alpha = 0.7                   # Learning rate
gamma = 0.618                 # Discounting rate

# EXPLORATION / EXPLOITATION PARAMETERS
max_epsilon = 1               # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability 
decay_rate = 0.01             # Exponential decay rate for exploration prob

# DEFINE FUNCTIONS
def greedy_policy(state):
    ''' Choose an action using the greedy policy '''
    return np.argmax(Q[state, :])  

def epsilon_policy(state, epsilon):
    ''' Choose an action using the epsilon policy '''
    exp_exp_tradeoff = random.uniform(0, 1)
    if exp_exp_tradeoff <= epsilon:
        action = env.action_space.sample()  # exploration
    else:
        action = np.argmax(Q[state, :])     # exploitation
    return action

def update_q1(current_state, action, reward, new_state, alpha):
    ''' Update the Q matrix with the Bellman equation (no learning rate, no discount factor) '''
    Q[current_state][action] += reward + np.max(Q[new_state, :])

def update_q2(current_state, action, reward, new_state, alpha):
    ''' Update the Q matrix with the Bellman equation (no learning rate, with discount factor) '''
    Q[current_state][action] += reward + gamma * np.max(Q[new_state, :])

def update_q3(current_state, action, reward, new_state, alpha):
    ''' Update the Q matrix with the Bellman equation (with learning rate and discount factor) '''
    Q[current_state][action] += alpha * (reward + gamma * np.max(Q[new_state]) - Q[current_state][action])

def get_epsilon(episode):
    ''' 
    Decrease the exploration rate at each episode 
    (less and less exploration is required as the agent learns)
    '''
    return min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)


# TRAINING PHASE
training_rewards = []

for episode in range(train_episodes):
    state = env.reset()
    episode_rewards = 0
    
    epsilon = get_epsilon(episode)

    for t in range(n_steps):
        action = epsilon_policy(state, epsilon)
        new_state, reward, done, info = env.step(action)
        # update_q1(state, action, reward, new_state, alpha)
        # update_q2(state, action, reward, new_state, alpha)
        update_q3(state, action, reward, new_state, alpha)
        episode_rewards += reward       
        state = new_state         # Update the state
        
        if done:
            print('Episode:{}/{} finished after {} timesteps | Total reward: {}'.format(episode, train_episodes, t+1, episode_rewards))
            break

    training_rewards.append(episode_rewards)

mean_train_reward = sum(training_rewards) / len(training_rewards)
print ("Average cumulated rewards over {} episodes: {}".format(train_episodes, mean_train_reward))


# PLOT REWARDS
x = range(train_episodes)
plt.plot(x, training_rewards)
plt.xlabel('Episode')
plt.ylabel('Training cumulative reward')
plt.savefig('plots/Q_learning_taxi.png', dpi=300)
plt.show()
    

# TEST PHASE
test_rewards = []

for episode in range(test_episodes):
    state = env.reset()
    episode_rewards = 0
    print("****************************************************")
    print("EPISODE ", episode)

    for t in range(n_steps):
        env.render()
        action = greedy_policy(state)       
        new_state, reward, done, info = env.step(action)
        episode_rewards += reward
        state = new_state
       
        if done:
            print('Episode:{}/{} finished after {} timesteps | Total reward: {}'.format(episode, test_episodes, t+1, episode_rewards))
            break

    test_rewards.append(episode_rewards)

mean_test_reward = sum(test_rewards) / len(test_rewards)
print ("Average cumulated rewards over {} episodes: {}".format(test_episodes, mean_test_reward))

env.close()
