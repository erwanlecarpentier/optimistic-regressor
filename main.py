import numpy as np
import matplotlib.pyplot as plt
import torch

import plot
import utils.config_reader as cfgr
import utils.trainer as tr

cfg = {
    'in_dim': 1,
    'out_dim': 1,
    'function_name': 'quadratic',
    'x_min': 0.0,  # Only for sampling domain, real domain is [0, 1]
    'x_max': 0.5,  # Only for sampling domain, real domain is [0, 1]
    'batch_size': 1000,
    'model_name': 'three_layers_nn',  # three_layers_nn, n_layers_nn
    'is_optimistic': False,
    'n_layers': 2,  # Parameter for n_layers_nn
    'h_dim': [10, 10],  # Hidden layer(s) dimension(s)
    'activation': 'relu',  # sigmoid, relu
    'learning_rate': 1e-3,
    'n_pass': 10000,
    'ratio_uniform_input': 1.0,
    'data_type': torch.float32,
    'device': torch.device('cpu')
}


def optimistic_regressor_experiment():
    # Initialize model and data
    x, y, f = cfgr.data_from_config(cfg)
    model = cfgr.model_from_config(cfg)
    is_optimistic = cfg['is_optimistic']

    # Fit
    if is_optimistic:
        tr.optimistic_train(model, x, y, n_pass=cfg['n_pass'])
    else:
        tr.train(model, x, y, n_pass=cfg['n_pass'])

    # Plot
    plot.plot(x, y, f, model, cfg)


if __name__ == "__main__":
    optimistic_regressor_experiment()
