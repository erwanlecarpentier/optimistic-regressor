import numpy as np
import matplotlib.pyplot as plt
import torch

import plot
import config_reader as cfgr


cfg = {
    'in_dim': 1,
    'out_dim': 1,
    'function_name': 'quadratic',
    'batch_size': 10,
    'model_name': 'n_layers_nn',
    'n_layers': 3,
    'h_dim': [10, 10],
    'activation': 'sigmoid',  # sigmoid, relu
    'learning_rate': 1e-2,
    'n_pass': 10000,
    'data_type': torch.float32,
    'device': torch.device('cpu')
}


def optimistic_regressor_experiment():
    # Initialize model and data
    x, y, f = cfgr.data_from_config(cfg)
    model = cfgr.model_from_config(cfg)

    # Fit
    model.train(x, y, n_pass=cfg['n_pass'])

    # Plot
    plot.plot_all(x, y, f, model, cfg)


if __name__ == "__main__":
    optimistic_regressor_experiment()
