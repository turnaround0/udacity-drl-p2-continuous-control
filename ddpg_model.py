import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as f

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, num_hidden_layers=(400, 300)):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            num_hidden_layers (list): list of number of nodes in hidden layers
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)

        """
        # Add the first layer, input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(state_size, num_hidden_layers[0])])

        # Add a variable number of more hidden layers
        layer_sizes = zip(num_hidden_layers[:-1], num_hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.output = nn.Linear(num_hidden_layers[-1], action_size)
        self.reset_parameters()
        """
        fc1_units=256
        fc2_units=128

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        """
        for hidden_layer in self.hidden_layers:
            hidden_layer.weight.data.uniform_(*hidden_init(hidden_layer))
        self.output.weight.data.uniform_(-3e-3, 3e-3)
        """
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        # Forward through each layer in `hidden_layers`, with leaky ReLU activation and dropout
        """
        x = state
        for linear in self.hidden_layers:
            x = f.leaky_relu(linear(x))

        return f.tanh(self.output(x))
        """
        x = f.relu(self.fc1(state))
        x = f.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, num_hidden_layers=(400, 300)):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            num_hidden_layers (list): list of number of nodes in hidden layers
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)

        """
        # Add the first layer, input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(state_size, num_hidden_layers[0])])
        
        # Add a variable number of more hidden layers
        num_copied_hidden_layers = list(num_hidden_layers)
        num_copied_hidden_layers[0] += action_size
        layer_sizes = zip(num_copied_hidden_layers[:-1], num_copied_hidden_layers[1:])

        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.output = nn.Linear(num_hidden_layers[-1], 1)
        self.reset_parameters()
        """
        fc1_units=256
        fc2_units=128
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        """
        for hidden_layer in self.hidden_layers:
            hidden_layer.weight.data.uniform_(*hidden_init(hidden_layer))
        self.output.weight.data.uniform_(-3e-3, 3e-3)
        """
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Forward through each layer in `hidden_layers`, with leaky ReLU activation and dropout
        # print(action) # TODO: Second actions are all zeros: too strange
        """
        x = f.leaky_relu(self.hidden_layers[0](state))
        x = torch.cat((x, action.type(torch.cuda.FloatTensor)), dim = 1)

        for linear in self.hidden_layers[1:]:
            x = f.leaky_relu(linear(x))

        return self.output(x)
        """
        xs = f.leaky_relu(self.fc1(state))
        x = torch.cat((xs, action.type(torch.cuda.FloatTensor)), dim=1)
        x = f.leaky_relu(self.fc2(x))
        return self.fc3(x)

