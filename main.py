import numpy as np
import matplotlib.pyplot as plt

from plot import plot_all
import functions
from models.n_layers_nn import NLayersNN


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


def data_from_config(config):
    if config['function_name'] == 'quadratic':
        in_dim = config['in_dim']
        batch_size = config['batch_size']
        out_dim = config['out_dim']

        # TODO handle out_dim in this case
        assert in_dim == out_dim

        x = np.random.rand(in_dim, batch_size)
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

        return NLayersNN(n_layers, in_dim, h_dim, out_dim, learning_rate)


def optimistic_regressor_experiment():
    # Initialize model and data
    model = model_from_config(cfg)
    x, y, f = data_from_config(cfg)

    # Fit

    # Plot
    plot_all(x, y, f, model)


if __name__ == "__main__":
    optimistic_regressor_experiment()
