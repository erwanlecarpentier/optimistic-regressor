"""
N layers optimistic neural network
"""

import numpy as np
import torch

from models.nnn import NLayersNN


class OptimisticNLayersNN(NLayersNN):

    def __init__(self, n_layers, in_dim, h_dim, out_dim, learning_rate, activation, data_type, device,
                 ratio_optimistic_points, optimistic_x_sampling_method):
        """
        :param in_dim: (int) input dimension
        :param h_dim: (int) hidden dimension
        :param out_dim: (int) output dimension
        """

        super().__init__(n_layers, in_dim, h_dim, out_dim, learning_rate, activation, data_type, device)

        self.ratio_optimistic_points = ratio_optimistic_points
        self.optimistic_x_sampling_method = optimistic_x_sampling_method

    def generate_optimistic_data(self, x, y):
        print('x: ', x.size())
        print('y: ', y.size())

        n_training_data = list(x.size())[0]
        n_optimistic_data = int(self.ratio_optimistic_points * list(x.size())[0])

        print('n_training_data   : ', n_training_data)
        print('n_optimistic_data : ', n_optimistic_data)

        # Generate input
        if self.optimistic_x_sampling_method == 'random':
            x_opt = torch.rand(n_optimistic_data, self.in_dim, device=self.device, dtype=self.data_type)

        # Generate optimistic output
        y_opt = torch.ones(n_optimistic_data, self.out_dim)

        print(x_opt.size())
        print(x_opt)
        print(y_opt.size())
        print(y_opt)

        exit()
        return x_opt, y_opt

    def train(self, x, y, n_pass=1000, verbose=True):

        x_opt, y_opt = self.generate_optimistic_data(x, y)

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
