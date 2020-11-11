import torch

import plot
import utils.config_reader as cfg_handler
import utils.trainer as tr

CONFIG = {
    'in_dim': 1,
    'out_dim': 1,
    'function_name': 'csin',  # quadratic, csin
    'x_min': 0.0,  # Only for sampling domain, real domain is [0, 1]
    'x_max': 0.5,  # Only for sampling domain, real domain is [0, 1]
    'batch_size': 100,
    'model_name': 'n_layers_nn',  # three_layers_nn, n_layers_nn
    'is_optimistic': True,
    'n_layers': 2,  # Parameter for n_layers_nn
    'h_dim': [20, 20, 20, 20],  # Hidden layer(s) dimension(s)
    'activation': 'relu',  # sigmoid, relu
    'learning_rate': 1e-3,
    'n_pass': 10000,
    'alpha': 0.1,
    'ratio_uniform_input': 1.0,
    'data_type': torch.float32,
    'device': torch.device('cpu')
}


def optimistic_regressor_experiment(config):
    # Initialize model and data
    x, y, f = cfg_handler.data_from_config(config)
    model = cfg_handler.model_from_config(config)
    is_optimistic = config['is_optimistic']

    # Fit
    if is_optimistic:
        tr.optimistic_train(model, x, y, n_pass=config['n_pass'],
                            alpha=config['alpha'],
                            ratio_uniform_input=config['ratio_uniform_input'],
                            lr=config['learning_rate'])
    else:
        tr.train(model, x, y, n_pass=config['n_pass'], lr=config['learning_rate'])

    # Plot
    plot.plot(x, y, f, model, config, show=False, export=True)


def grid_search(config):
    ratios = [0.1, 0.5, 1.0]

    for r in ratios:
        CONFIG['ratio_uniform_input'] = r
        optimistic_regressor_experiment(config)


if __name__ == "__main__":
    # optimistic_regressor_experiment(CONFIG)
    grid_search(CONFIG)
