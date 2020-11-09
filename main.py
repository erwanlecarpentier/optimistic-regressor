import numpy as np
import matplotlib.pyplot as plt

import plot
import config_reader as cfgr


cfg = {
    'in_dim': 1,
    'out_dim': 1,
    'function_name': 'quadratic',
    'batch_size': 5,
    'model_name': 'n_layers_nn',
    'n_layers': 2,
    'h_dim': [1, 2],
    'learning_rate': 1e-6
}


def optimistic_regressor_experiment():
    # Initialize model and data
    model = cfgr.model_from_config(cfg)
    x, y, f = cfgr.data_from_config(cfg)

    # Fit

    # Plot
    plot.plot_all(x, y, f, model)


if __name__ == "__main__":
    optimistic_regressor_experiment()
