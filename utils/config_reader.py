import numpy as np
import torch

import functions
from models.nnn import NLayersNN
from models.onnn import OptimisticNLayersNN


def data_from_config(config):
    if config['function_name'] == 'quadratic':
        in_dim = config['in_dim']
        x_min = config['x_min']
        x_max = config['x_max']
        batch_size = config['batch_size']
        out_dim = config['out_dim']
        data_type = config['data_type']
        device = config['device']

        # TODO handle out_dim in this case
        assert in_dim == out_dim

        x = torch.rand(batch_size, in_dim, device=device, dtype=data_type) * (x_max - x_min) + x_min
        f = functions.Quadratic()
        y = f.forward(x)

        return x, y, f


def model_from_config(config):
    n_layers = config['n_layers']
    in_dim = config['in_dim']
    h_dim = config['h_dim']
    out_dim = config['out_dim']
    learning_rate = config['learning_rate']
    activation = config['activation']
    data_type = config['data_type']
    device = config['device']

    if config['model_name'] == 'nnn':
        return NLayersNN(n_layers=n_layers, in_dim=in_dim, h_dim=h_dim, out_dim=out_dim,
                         learning_rate=learning_rate, activation=activation, data_type=data_type,
                         device=device)

    if config['model_name'] == 'onnn':
        ratio_uniform_input = config['ratio_uniform_input']

        return OptimisticNLayersNN(n_layers=n_layers, in_dim=in_dim, h_dim=h_dim, out_dim=out_dim,
                                   learning_rate=learning_rate, activation=activation,
                                   data_type=data_type, device=device,
                                   ratio_uniform_input=ratio_uniform_input)
