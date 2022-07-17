import numpy as np
import matplotlib.pyplot as plt
import gym


# INITIALISE THE ENVIRONMENT
env = gym.make("Taxi-v3")
action_size = env.action_space.n
state_size = env.observation_space.n
print("Action space size: ", action_size)
print("State space size: ", state_size)

# INITIALISE Q TABLE TO ZERO
Q = np.zeros((state_size, action_size))

# HYPERPARAMETERS
n_episodes = 2000             # Total train episodes
n_steps = 100                 # Max steps per episode
alpha = 0.7                   # Learning rate
gamma = 0.618                 # Discounting rate

# EXPLORATION / EXPLOITATION PARAMETERS
epsilon = 1                   # Exploration rate
max_epsilon = 1               # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability 
decay_rate = 0.01             # Exponential decay rate for exploration prob

# TRAINING PHASE
rewards = []

for episode in range(n_episodes):
    state = env.reset()
    episode_rewards = 0
    
    for t in range(n_steps):
        # Choose an action greedily 
        action = np.argmax(Q[state,:]) 

        # Perform the action 
        new_state, reward, done, info = env.step(action)
        
        # Update the Q matrix using the Bellman equation: Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        Q[state, action] = Q[state, action] + reward + np.max(Q[new_state, :]) 

        episode_rewards += reward  # increment the cumulative reward        
        state = new_state          # Update the state
        
        # If we reach the end of the episode
        if done:
            print('Episode:{}/{} finished after {} timesteps | Total reward: {}'.format(episode, n_episodes, t+1, episode_rewards))
            break
    
    # append the episode cumulative reward to the reward list
    rewards.append(episode_rewards)

mean_cum_reward = sum(rewards) / len(rewards)
print ("Average cumulated rewards over {} episodes: {}".format(n_episodes, mean_cum_reward))

env.close()


# PLOT REWARDS
x = range(n_episodes)
plt.plot(x, rewards)
plt.xlabel('Episode')
plt.ylabel('Training cumulative reward')
plt.savefig('plots/Q_learning_simple_update.png', dpi=300)
plt.show()
