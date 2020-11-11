import numpy as np
import torch

import functions
from models.n_layers_nn import NLayersNN
from models.three_layers_nn import ThreeLayersNN


def data_from_config(config):
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

    if config['function_name'] == 'csin':
        f = functions.CSin()
    else:
        f = functions.Quadratic()

    y = f.forward(x)

    return x, y, f


def model_from_config(config):
    in_dim = config['in_dim']
    h_dim = config['h_dim']
    out_dim = config['out_dim']
    activation = config['activation']

    if config['model_name'] == 'three_layers_nn':
        return ThreeLayersNN(in_dim=in_dim, out_dim=out_dim, h_dim=h_dim,activation=activation)

    if config['model_name'] == 'n_layers_nn':
        return NLayersNN(in_dim=in_dim, h_dim=h_dim, out_dim=out_dim,
                         activation=activation)
