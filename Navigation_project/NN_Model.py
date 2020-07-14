# import framework

import torch
import torch.nn as nn
import torch.nn.functional as F

# Build model

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=32, fc3_units=16):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            fc1_units (int): Dimension of the first hidden/fully connected layer
            fc2_units (int): Dimension of the second hidden/fully connected layer
            fc3_units (int): Dimension of the third hidden/fully connected layer
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, action_size)
        
    def forward(self, state):
        """Build a network that maps state -> action values."""
        action_values = F.relu(self.fc1(state))
        action_values = F.relu(self.fc2(action_values))
        action_values = F.relu(self.fc3(action_values))
        action_values = self.fc4(action_values)
        return action_values
  
