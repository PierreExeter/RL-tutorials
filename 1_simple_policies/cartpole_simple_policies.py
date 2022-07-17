import gym


# INITIALISE THE ENVIRONMENT
env = gym.make('CartPole-v1', render_mode='human')

# PRINT ENV ACTION AND OBSERVATION SPACE
print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)


# DEFINE POLICY
def policy1():
    """
    This policy returns a random action sampled in the action space.
    """
    return env.action_space.sample()  

def policy2(t):
    """ 
    This policy returns action 0 (push left) during the first 
    20 timesteps and action 1 (push right) during the remaining
    timesteps. 
    """
    action = 0
    if t < 20:
        action = 0
    else:
        action = 1
    return action

def policy3(t):
    """ 
    This policy alternates between action 0 (push left) 
    if the timestep number is odd and action 1 (push right) otherwise. 
    """
    action = 0
    if t%2 == 1:
        action = 1
    return action


nb_episodes = 20
nb_timesteps = 100
policy_nb = 3   # choose policy number here
cum_rewards = []

# ITERATE OVER EPISODES
for episode in range(nb_episodes):
    state = env.reset()
    episode_rewards = 0
    
    # ITERATE OVER TIME-STEPS
    for t in range(nb_timesteps):

        if policy_nb == 1:
            action = policy1()
        elif policy_nb == 2:
            action = policy2(t)
        elif policy_nb == 3:
            action = policy3(t)

        state, reward, done, info = env.step(action)
        episode_rewards += reward
        
        if done: 
            print('Episode:{}/{} finished after {} timesteps | Total reward: {}'.format(episode, nb_episodes, t+1, episode_rewards))
            break

    # Append the episode cumulative reward to the reward list    
    cum_rewards.append(episode_rewards)

mean_cum_reward = sum(cum_rewards) / len(cum_rewards)
print("The mean of the cumulative rewards over {} episodes for policy {} is: {}".format(nb_episodes, policy_nb, mean_cum_reward))

env.close()
