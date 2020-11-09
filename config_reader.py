import numpy as np
import torch

import functions
from models.n_layers_nn import NLayersNN


def data_from_config(config):
    if config['function_name'] == 'quadratic':
        in_dim = config['in_dim']
        batch_size = config['batch_size']
        out_dim = config['out_dim']
        data_type = config['data_type']
        device = config['device']

        # TODO handle out_dim in this case
        assert in_dim == out_dim

        x = torch.rand(batch_size, in_dim, device=device, dtype=data_type)
        f = functions.Quadratic()
        y = f.predict(x)

        return x, y, f


def model_from_config(config):
    if config['model_name'] == 'n_layers_nn':

        n_layers = config['n_layers']
        in_dim = config['in_dim']
        h_dim = config['h_dim']
        out_dim = config['out_dim']
        learning_rate = config['learning_rate']
        data_type = config['data_type']
        device = config['device']

        return NLayersNN(n_layers, in_dim, h_dim, out_dim, learning_rate, data_type, device)
