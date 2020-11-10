"""
N layers optimistic neural network
"""

import numpy as np
import torch

from models.nnn import NLayersNN


class OptimisticNLayersNN(NLayersNN):

    def __init__(self, n_layers, in_dim, h_dim, out_dim, learning_rate, activation, data_type, device,
                 ratio_uniform_input):
        """
        :param in_dim: (int) input dimension
        :param h_dim: (int) hidden dimension
        :param out_dim: (int) output dimension
        """

        super().__init__(n_layers, in_dim, h_dim, out_dim, learning_rate, activation, data_type, device)

        self.ratio_uniform_input = ratio_uniform_input

    def generate_uniform_input(self, x):
        n_training_data = list(x.size())[0]
        n_optimistic_data = int(self.ratio_uniform_input * n_training_data)
        return torch.rand(n_optimistic_data, self.in_dim, device=self.device, dtype=self.data_type)

    def generate_optimistic_data(self, x):
        n_training_data = list(x.size())[0]
        n_optimistic_data = int(self.ratio_uniform_input * n_training_data)

        # Generate input
        if self.ratio_uniform_input == 'random':
            x_opt = torch.rand(n_optimistic_data, self.in_dim, device=self.device, dtype=self.data_type)
        # Generate optimistic output
        y_opt = torch.ones(n_optimistic_data, self.out_dim)

        return x_opt, y_opt

    def forward(self, x):
        h_act = x  # tmp
        for i in range(self.n_layers - 1):
            h = h_act.mm(self.w['w' + str(i)])

            if self.activation == 'relu':
                h_act = h.clamp(min=0)
            if self.activation == 'sigmoid':
                h_act = torch.sigmoid(h)

        return 1.0 - h_act.mm(self.w['w' + str(self.n_layers - 1)])

    def my_train(self, x, y, n_pass=1000, verbose=True):

        x_u = self.generate_uniform_input(x)

        for t in range(n_pass):
            # Predict
            y_pred = self.forward(x)

            # Compute loss
            # loss = (y - y_pred).pow(2).sum()
            loss = ((1.0 - y_pred - y) - (1.0 - y_pred)).pow(2).sum()

            '''Second pass'''
            # Predict
            y_pred = self.forward(x_u)

            # Compute loss
            loss += (1.0 - y_pred).pow(2).sum()
            ''''''

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
