"""
N layers optimistic neural network
"""

import numpy as np
import torch

from models.nnn import NLayersNN


class OptimisticNLayersNN(NLayersNN):

    def __init__(self, n_layers, in_dim, h_dim, out_dim, learning_rate, activation, data_type, device):
        """
        :param in_dim: (int) input dimension
        :param h_dim: (int) hidden dimension
        :param out_dim: (int) output dimension
        """

        super().__init__(n_layers, in_dim, h_dim, out_dim, learning_rate, activation, data_type, device)

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
