import numpy as np
import torch
import matplotlib.pyplot as plt


def plot_all(x, y, f, model):
    # TODO handle higher dimensions

    x_ref = np.arange(0.0, 1.05, 0.05, dtype=float)
    y_ref = f.predict(torch.from_numpy(x_ref))
    y_pred = model.predict(x_ref)

    fig, ax = plt.subplots()

    # ax.plot(x_ref, y_pred, 'darkorange', label='Model')
    ax.plot(x_ref, y_ref.numpy(), 'teal', label='True function')
    ax.scatter(x.numpy(), y.numpy(), c='teal', label='Data', alpha=1.0, edgecolors='none')

    '''
    handles, labels = ax.get_legend_handles_labels()
    handles = [handles[0], handles[2], handles[1]]
    labels = [labels[0], labels[2], labels[1]]
    ax.legend(handles, labels, loc=2)
    '''

    ax.axis([-.05, 1.05, -.05, 1.05])
    ax.set_xlabel('Input')
    ax.set_ylabel('Output')
    ax.grid(True)

    plt.show()
