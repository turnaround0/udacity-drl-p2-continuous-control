import torch
import torch.nn as nn
import torch.nn.functional as f


class QNetwork(nn.Module):
    """
    Actor (Policy) Model.
    """

    def __init__(self, state_size, action_size, seed, hidden_layers=(64, 64), drop_p=0.5):
        """
        Initialize parameters and build model.

        :param state_size (int): Dimension of each state
        :param action_size (int): Dimension of each action
        :param seed (int): Random seed
        :param hidden_layers (list): list of number of nodes in hidden layers
        :param drop_p (float): probability of dropout
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        # Add the first layer, input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(state_size, hidden_layers[0])])

        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.output = nn.Linear(hidden_layers[-1], action_size)
        self.dropout = nn.Dropout(p=drop_p)

    def forward(self, state):
        """
        Build a network that maps state -> action values.

        :param state: state values
        :return action values
        """

        # Forward through each layer in `hidden_layers`, with leaky ReLU activation and dropout
        x = state
        for linear in self.hidden_layers:
            x = f.leaky_relu(linear(x))
            x = self.dropout(x)

        return self.output(x)
