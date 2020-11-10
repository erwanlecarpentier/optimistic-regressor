"""
N layers optimistic neural network
"""

import torch


class ThreeLayersNN(torch.nn.Module):

    def __init__(self, in_dim, out_dim):
        """
        :param in_dim: (int) input dimension
        :param out_dim: (int) output dimension
        """
        super(ThreeLayersNN, self).__init__()

        # Initialize layers
        s = 10
        self.linear1 = torch.nn.Linear(in_dim, s)
        self.linear2 = torch.nn.Linear(s, s)
        self.linear3 = torch.nn.Linear(s, out_dim)

    def forward(self, x):
        h = self.linear1(x).clamp(min=0)
        h = self.linear2(h).clamp(min=0)
        return self.linear3(h)
