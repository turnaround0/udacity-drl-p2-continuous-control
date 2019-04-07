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

        # Add the first layer, input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(state_size, num_hidden_layers[0])])

        # Add a variable number of more hidden layers
        layer_sizes = zip(num_hidden_layers[:-1], num_hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.output = nn.Linear(num_hidden_layers[-1], action_size)
        self.reset_parameters()

    def reset_parameters(self):
        for hidden_layer in self.hidden_layers:
            hidden_layer.weight.data.uniform_(*hidden_init(hidden_layer))
        self.output.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        # Forward through each layer in `hidden_layers`, with leaky ReLU activation and dropout
        x = state
        for linear in self.hidden_layers:
            x = f.leaky_relu(linear(x))

        return f.tanh(self.output(x))

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

        # Add the first layer, input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(state_size, num_hidden_layers[0])])
        
        # Add a variable number of more hidden layers
        num_copied_hidden_layers = list(num_hidden_layers)
        num_copied_hidden_layers[0] += action_size
        layer_sizes = zip(num_copied_hidden_layers[:-1], num_copied_hidden_layers[1:])

        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.output = nn.Linear(num_hidden_layers[-1], 1)
        self.reset_parameters()
        
    def reset_parameters(self):
        for hidden_layer in self.hidden_layers:
            hidden_layer.weight.data.uniform_(*hidden_init(hidden_layer))
        self.output.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Forward through each layer in `hidden_layers`, with leaky ReLU activation and dropout
        x = f.leaky_relu(self.hidden_layers[0](state))
        x = torch.cat((x, action.type(torch.FloatTensor)), dim = 1)

        for linear in self.hidden_layers[1:]:
            x = f.leaky_relu(linear(x))

        return self.output(x)
