"""
N layers optimistic neural network
"""

import torch


class ThreeLayersNN(torch.nn.Module):

    def __init__(self, in_dim, out_dim, h_dim, activation='relu'):
        """
        :param in_dim: (int) input dimension
        :param out_dim: (int) output dimension
        """
        super(ThreeLayersNN, self).__init__()
        self.activation = activation

        # Initialize layers
        self.h_dim = h_dim
        self.linear1 = torch.nn.Linear(in_dim, h_dim[0])
        self.linear2 = torch.nn.Linear(h_dim[0], h_dim[1])
        self.linear3 = torch.nn.Linear(h_dim[1], out_dim)

    def forward(self, x):
        if self.activation == 'sigmoid':
            h = torch.sigmoid(self.linear1(x))
            h = torch.sigmoid(self.linear2(h))
            return self.linear3(h)
        elif self.activation == 'relu':
            h = self.linear1(x).clamp(min=0)
            h = self.linear2(h).clamp(min=0)
            return self.linear3(h)
