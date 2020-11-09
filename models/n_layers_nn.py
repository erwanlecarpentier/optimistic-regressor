"""
Two layers neural network
"""

import torch


class NLayersNN(object):

    def __init__(self, n_layers, in_dim, h_dim, out_dim, learning_rate):
        """
        :param in_dim: (int) input dimension
        :param h_dim: (int) hidden dimension
        :param out_dim: (int) output dimension
        """
        data_type = torch.float
        device = torch.device("cpu")

        self.n_layers = n_layers
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.learning_rate = learning_rate

        for i in range(n_layers):
            print(i)
        # self.w1 = torch.randn(in_dim, h_dim, device=device, dtype=data_type)
        # self.w2 = torch.randn(h_dim, out_dim, device=device, dtype=data_type)

    def predict(self, x):
        # TODO
        return x + self.learning_rate
