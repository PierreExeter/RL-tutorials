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
epsilon = 1                   # Exploration rate
max_epsilon = 1               # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability 
decay_rate = 0.01             # Exponential decay rate for exploration prob

# TRAINING PHASE
training_rewards = []

for episode in range(train_episodes):
    state = env.reset()
    cumulative_training_rewards = 0
    
    # EPSILON POLICY
    for t in range(n_steps):
        # Choose an action (a) among the possible states (s)
        exp_exp_tradeoff = random.uniform(0, 1)   # choose a random number
        
        # If this number > epsilon, select the action corresponding to the biggest Q value for this state (Exploitation)
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(Q[state,:])        
        # Else choose a random action (Exploration)
        else:
            action = env.action_space.sample()
        
        # Perform the action (a) and observe the outcome state(s') and reward (r)
        new_state, reward, done, info = env.step(action)

        # Update the Q table using the Bellman equation: Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[new_state, :]) - Q[state, action]) 
        cumulative_training_rewards += reward  # increment the cumulative reward        
        state = new_state         # Update the state
        
        # If we reach the end of the episode
        if done:
            print('Episode:{}/{} finished after {} timesteps | Total reward: {}'.format(episode, train_episodes, t+1, cumulative_training_rewards))
            break
    
    # Reduce epsilon (because we need less and less exploration)
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
    
    # append the episode cumulative reward to the list
    training_rewards.append(cumulative_training_rewards)

mean_train_reward = sum(training_rewards) / len(training_rewards)
print ("Average cumulated rewards over {} episodes: {}".format(train_episodes, mean_train_reward))


# PLOT REWARDS
x = range(train_episodes)
plt.plot(x, training_rewards)
plt.xlabel('Episode')
plt.ylabel('Training cumulative reward')
plt.savefig('plots/Q-learning_discounted_reward_learning_rate_epsilon_policy.png', dpi=300)
plt.show()
    

# TEST PHASE
test_rewards = []

for episode in range(test_episodes):
    state = env.reset()
    cumulative_test_rewards = 0
    print("****************************************************")
    print("EPISODE ", episode)

    for t in range(n_steps):
        env.render()
        # Take the action (index) that have the maximum expected future reward given that state
        # (greedy policy)
        action = np.argmax(Q[state,:])        
        new_state, reward, done, info = env.step(action)
        cumulative_test_rewards += reward
        state = new_state
       
        if done:
            print('Episode:{}/{} finished after {} timesteps | Total reward: {}'.format(episode, test_episodes, t+1, cumulative_test_rewards))
            break

    test_rewards.append(cumulative_test_rewards)

mean_test_reward = sum(test_rewards) / len(test_rewards)
print ("Average cumulated rewards over {} episodes: {}".format(test_episodes, mean_test_reward))

env.close()
