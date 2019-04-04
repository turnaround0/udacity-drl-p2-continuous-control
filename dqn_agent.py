import numpy as np
import random

import torch
import torch.nn.functional as f
import torch.optim as optim

from model import QNetwork


class DQNAgent:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size,
                 memory=None, device='cpu', weights_filename=None, params=None, train_mode=True):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            memory (obj): Memory buffer to sample
            device (str): device string between cuda:0 and cpu
            weights_filename (str): file name having weights of local Q network to load
            params (dict): hyper-parameters
            train_mode (bool): True if it is train mode, otherwise False
        """
        self.state_size = state_size
        self.action_size = action_size
        self.device = device

        # Set parameters
        self.gamma = params['gamma']
        self.tau = params['tau']
        self.lr = params['lr']
        self.update_every = params['update_every']
        self.seed = random.seed(params['seed'])

        # Q-Network
        if train_mode:
            drop_p = params['drop_p']
        else:
            drop_p = 0

        self.qnetwork_local = QNetwork(state_size, action_size, params['seed'],
                                       params['hidden_layers'], drop_p).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, params['seed'],
                                        params['hidden_layers'], drop_p).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)

        # Replay memory
        self.memory = memory

        # Load weight file
        if weights_filename:
            self.qnetwork_local.load_state_dict(torch.load(weights_filename))

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def store_weights(self, filename):
        """Store weights of Q local network

        Params
        ======
            filename (str): string of filename to store weights of Q local network
        """
        torch.save(self.qnetwork_local.state_dict(), filename)

    def step(self, state, action, reward, next_state, done):
        """This defines an agent to do whenever moving.

        Params
        ======
            state (array_like): current state
            action (int): current action
            reward (float): reward on next state
            next_state (array_like): next state
            done (bool): flag to indicate whether this episode is done
        """
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.memory.get_batch_size():
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        # q_targets_next = max Q(next state, next action, theta dash)
        # qnetwork_target(next_states): Q values[next_states][action]
        #   .detach(): detached from the current graph
        #   .max(1): first - max(Q values), second - action: argmax(Q values), third - device (cuda:0 or cpu)
        #   .max(1)[0]: select max(Q values) of next states
        #   .unsqueeze(1)): from 1d-array to 2d-matrix ([a,b,c] -> [[a], [b], [c]])
        q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        # Compute Q targets for current states
        # If done, q_targets = rewards.
        # Otherwise, q_targets = rewards + gamma * q_targets_next
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))

        # Get expected Q values from local model
        q_locals = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = f.mse_loss(q_locals, q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()  # Clear gradients
        loss.backward()             # Calculate gradients
        self.optimizer.step()       # Move to the gradients

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    @staticmethod
    def soft_update(local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0 - tau) * target_param.data)
