"""
N layers optimistic neural network
"""

import numpy as np
import torch


class NLayersNN(object):

    def __init__(self, n_layers, in_dim, h_dim, out_dim, learning_rate, data_type, device):
        """
        :param in_dim: (int) input dimension
        :param h_dim: (int) hidden dimension
        :param out_dim: (int) output dimension
        """

        self.n_layers = n_layers
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.learning_rate = learning_rate
        self.w = {}

        sizes = np.concatenate(([in_dim], h_dim, [out_dim]), axis=0)
        for i in range(n_layers):
            self.w['w' + str(i)] = torch.randn(sizes[i], sizes[i+1], device=device, dtype=data_type, requires_grad=True)

    def predict(self, x):
        h_relu = x  # tmp
        for i in range(self.n_layers - 1):
            h = h_relu.mm(self.w['w' + str(i)])
            h_relu = h.clamp(min=0)

        return h_relu.mm(self.w['w' + str(self.n_layers - 1)])

    def train(self, x, y, n_pass=1000, verbose=True):
        for t in range(n_pass):
            # Predict
            y_pred = self.predict(x)

            # Compute loss
            loss = (y_pred - y).pow(2).sum()
            if verbose and t % 100 == 99:
                print('Iteration: ', t+1, '  loss:', loss.item())

            # Backward pass
            loss.backward()

            # Update weights
            with torch.no_grad():
                for i in range(self.n_layers):
                    self.w['w' + str(i)] -= self.learning_rate * self.w['w' + str(i)].grad

                    # Manually zero the gradients after updating weights
                    self.w['w' + str(i)].grad.zero_()
