"""
N layers optimistic neural network
"""

import numpy as np
import torch


class NLayersNN(torch.nn.Module):

    def __init__(self, n_layers, in_dim, h_dim, out_dim, activation):
        """
        :param in_dim: (int) input dimension
        :param h_dim: (int) hidden dimension
        :param out_dim: (int) output dimension
        """
        super(NLayersNN, self).__init__()

        self.n_layers = n_layers
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.activation = activation

        # Initialize layers
        sizes = np.concatenate(([in_dim], h_dim, [out_dim]), axis=0)
        self.layers = torch.nn.ModuleList([torch.nn.Linear(sizes[i], sizes[i + 1])
                                           for i in range(n_layers + 1)])

    def forward(self, x):
        if self.activation == 'sigmoid':
            for i in range(self.n_layers):
                x = torch.sigmoid(self.layers[i](x))
            return self.layers[self.n_layers](x)
        elif self.activation == 'relu':
            for i in range(self.n_layers):
                x = self.layers[i](x).clamp(min=0)
            return self.layers[self.n_layers](x)
