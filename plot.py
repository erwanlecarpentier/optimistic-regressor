import numpy as np
import torch
import matplotlib.pyplot as plt


def plot_all(x, y, f, model, config):
    data_type = config['data_type']

    # TODO handle higher dimensions

    x_ref = np.arange(0.0, 1.05, 0.05, dtype=float)
    x_ref = x_ref.reshape((x_ref.shape[0], 1))
    x_ref_as_tensor = torch.from_numpy(x_ref)
    x_ref_as_tensor = x_ref_as_tensor.type(data_type)

    y_ref = f.forward(x_ref_as_tensor)
    y_pred = model.forward(x_ref_as_tensor)

    fig, ax = plt.subplots()

    ax.plot(x_ref, y_pred.detach().numpy(), 'darkorange', label='Model')
    ax.plot(x_ref, y_ref.numpy(), 'teal', label='True function')
    ax.scatter(x.numpy(), y.numpy(), c='teal', label='Data', alpha=1.0, edgecolors='none')

    handles, labels = ax.get_legend_handles_labels()
    handles = [handles[0], handles[2], handles[1]]
    labels = [labels[0], labels[2], labels[1]]
    ax.legend(handles, labels, loc=2)

    ax.axis([-.05, 1.05, -.05, 1.05])
    ax.set_xlabel('Input')
    ax.set_ylabel('Output')
    ax.grid(True)

    plt.show()
