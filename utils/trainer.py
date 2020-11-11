import torch


def test_loss(y_pred, y):
    return (y_pred - 1.0).pow(2).sum()


def my_mse_loss(y_pred, y):
    return (y_pred - y).pow(2).sum()


def optimistic_loss(y_pred, y, y_uniform):
    return ((y_pred - (1.0 - y)).pow(2) - y_pred.pow(2)).sum() + y_uniform.pow(2).sum()


def generate_uniform_input(x, ratio_uniform_input=1.0):
    n_input, input_dimension = tuple(x.size())
    n_optimistic_data = int(ratio_uniform_input * n_input)
    return torch.rand(n_optimistic_data, input_dimension, device=x.device, dtype=x.dtype)


def optimistic_train(model, x, y, n_pass, ratio_uniform_input, lr=1e-4):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    x_uniform = generate_uniform_input(x, ratio_uniform_input=ratio_uniform_input)
    alpha = 0.1

    for t in range(n_pass):
        # Forward pass
        y_pred = model(x)
        y_uniform = model(x_uniform)

        # Compute and print loss
        # loss = optimistic_loss(y_pred, y, y_uniform)
        loss = ((y_pred - (1.0 - y)).pow(2) - alpha * y_pred.pow(2)).sum() + alpha * y_uniform.pow(2).sum()

        if t % 100 == 99:
            print(t+1, loss.item())

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def train(model, x, y, n_pass, criterion='mse', lr=1e-4):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    for t in range(n_pass):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x)

        # Compute and print loss
        if criterion == 'mse':
            loss = my_mse_loss(y_pred, y)
        else:
            loss = test_loss(y_pred, y)

        if t % 100 == 99:
            print(t+1, loss.item())

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
