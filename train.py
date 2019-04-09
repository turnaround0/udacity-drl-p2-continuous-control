from unityagents import UnityEnvironment
import pandas as pd
import numpy as np

import time
import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from replaybuffer import ReplayBuffer
from ddpg_agent import DDPGAgent

reacher_filename = 'Reacher_Linux_One_NoVis/Reacher.x86_64'

memory_params = {
    'buffer_size': int(1e6),        # replay buffer size
    'batch_size': 128,              # minibatch size
    'seed': 0,                      # Seed to generate random numbers
}

params = {
    'gamma': 0.99,                      # discount factor
    'tau': 1e-3,                        # for soft update of target parameters
    'update_every': 10,                 # update parameters per this number
    'lr_actor': 3e-4,                   # learning rate of the Actor
    'lr_critic': 3e-4,                  # learning rate of the Critic
    'seed': 0,                          # Seed to generate random numbers
    'actor_units': [512, 256],          # Number of nodes in hidden layers of the Actor
    'critic_units': [512, 256],         # Number of nodes in hidden layers of the Critic
    'weight_decay': 0,                  # L2 weight decay
    'noise_theta': 0.15,                # Theta of Ornstein-Uhlenbeck process
    'noise_sigma': 0.01,                # Sigma of Ornstein-Uhlenbeck process
}

# Parameters to store and plot scores
rolling_n_episodes = 10     # Score is checked whenever number of tries reachs to this.
benchmark_score = 30.0      # Score of agent should be over this score


def train(n_episodes=400, max_t=5000, agents=None, filenames=None,
          benchmark_score=30, rolling_n_episodes=10):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode (should be over 1000)
        agents (obj): training agent instances
        filenames (list): string of filenames to store weights of actor and critic
        benchmark_score (int): the score of agent should be over this score
        rolling_n_episodes (int): the score is checked whenever number of tries reachs to this
    """
    start_time = time.time()
        
    all_agent_scores = []                             # list containing scores from each episode for all agents
    scores_window = deque(maxlen=rolling_n_episodes)  # last rolling_n_episodes scores
    avg_checked = False
    
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        states = env_info.vector_observations              # get the current state (for each agent)
        scores = np.zeros(len(agents))                     # initialize the score (for each agent)

        for agent in agents:                               # Reset agent before starting new episode
            agent.reset()

        for t in range(max_t):
            actions = [agent.act(states[i]) for i, agent in enumerate(agents)]  # select actions
            env_info = env.step(actions)[brain_name]       # send the action to the environment
            next_states = env_info.vector_observations     # get the next state
            rewards = env_info.rewards                     # get the reward
            dones = env_info.local_done                    # see if episode has finished

            for i, agent in enumerate(agents):
                agent.step(states[i], actions[i], rewards[i], next_states[i], dones[i])
            
            states = next_states                           # roll over the state to next time step
            scores += rewards                              # update the score
            if np.any(dones):                              # exit loop if episode finished
                break

        avg_score = np.mean(scores)                        # average score of all agents
        scores_window.append(avg_score)                    # save most recent score
        all_agent_scores.append(avg_score)                 # save most recent score
        avg_scores_window = np.mean(scores_window)         # get average score of current window

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, avg_scores_window), end="")
        
        if i_episode % rolling_n_episodes == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, avg_scores_window))
        
        if not avg_checked and avg_scores_window >= benchmark_score:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
                  i_episode - rolling_n_episodes, avg_scores_window))
            avg_checked = True

    print('\nTraining time = {:.2f}s'.format(time.time() - start_time))
    if filenames:
        agent.store_weights(filenames)

    return all_agent_scores


def test(agents, max_t=5000):
    env_info = env.reset(train_mode=False)[brain_name] # reset the environment
    states = env_info.vector_observations              # get the current state
    scores = np.zeros(len(agents))

    for t in range(max_t):
        actions = [agent.act(states[i]) for i, agent in enumerate(agents)]  # select actions
        env_info = env.step(actions)[brain_name]       # send the action to the environment
        rewards = env_info.rewards                     # get the reward
        dones = env_info.local_done                    # see if episode has finished
        states = env_info.vector_observations          # roll over the state to next time step
        scores += rewards                              # update the score
        if np.any(dones):                              # exit loop if episode finished
            break
    
    print('Score: {:.2f}'.format(np.mean(scores)))


def plot_scores(scores, benchmark_score, rolling_n_episodes):
    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    ax.axhline(benchmark_score, c="red", alpha=0.5)
    
    rolling_window = rolling_n_episodes
    rolling_mean = pd.Series(scores).rolling(rolling_window).mean()
    plt.plot(rolling_mean, c='yellow', alpha=0.7);
    
    plt.show()

# Select environment of Reacher
env = UnityEnvironment(file_name=reacher_filename)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

# Set device between cuda:0 and cpu
torch_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device =', torch_device)

memory = ReplayBuffer(action_size, memory_params['buffer_size'],
                      memory_params['batch_size'], memory_params['seed'], torch_device)

ddpg_agents = [DDPGAgent(state_size, action_size, memory, torch_device, params)
               for _ in range(num_agents)]

ddpg_scores = train(400, 5000, ddpg_agents, ["model_ddpg_actor.pth", "model_ddpg_critic.pth"],
                    benchmark_score, rolling_n_episodes)

plot_scores(ddpg_scores, benchmark_score, rolling_n_episodes)

test(ddpg_agents)

env.close()
