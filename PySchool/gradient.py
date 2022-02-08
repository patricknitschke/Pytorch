from numpy import dtype


def numpy():
    import numpy as np

    # f = w * x  (f = 2*x)

    x = np.array([1,2,3,4], dtype=np.float64)
    y = np.array([2,4,6,8], dtype=np.float64)

    w = 0.0

    learning_rate = 0.1

    print(f"Prediction before training: f(5) = {(w*5):.3f}")
    for epoch in range(10):
        
        # Forward
        y_hat = w*x
        loss = ((y-y_hat)**2).mean()

        # Gradient
        grad_w = np.dot(2*x, y_hat-y).mean()

        # Update
        w -= learning_rate * grad_w
        
        # Log
        if epoch % 1 == 0:
            print(f"epoch {epoch+1}: w = {w:.3f}, loss = {loss:.8f}")

    print(f"Prediction after training: f(5) = {(w*5):.3f}")


def manual_torch():
    import torch as th
    # f = w * x  (f = 2*x)

    x = th.tensor([1,2,3,4], dtype=th.float32)
    y = th.tensor([1,2,3,4], dtype=th.float32)

    w = th.zeros(1, dtype=th.float32, requires_grad=True)

    learning_rate = 0.1

    print(f"Prediction before training: f(5) = {(w*5):.3f}")
    for epoch in range(50):
        
        # Forward
        y_hat = w*x
        loss = ((y-y_hat)**2).mean()

        # Gradient
        loss.backward()

        # Update
        with th.no_grad():
            w -= learning_rate * w.grad
        
        w.grad.zero_()
        
        # Log
        if epoch % 1 == 0:
            print(f"epoch {epoch+1}: w = {w.item():.3f}, loss = {loss:.8f}")
    
    print(f"Prediction after training: f(5) = {(w*5):.3f}")


def half_torch():
    import torch as th
    import torch.nn as nn

    x = th.tensor([1,2,3,4], dtype=th.float32)
    y = th.tensor([1,2,3,4], dtype=th.float32)

    w = th.zeros(1, dtype=th.float32, requires_grad=True)

    # Training
    learning_rate = 0.05
    epochs = 10

    loss = nn.MSELoss()
    optimiser = th.optim.SGD([w], lr=learning_rate)

    print(f"Prediction before training: f(5) = {(w*5):.3f}")
    for epoch in range(epochs):
        
        # Forward
        y_hat = w*x
        l = loss(y, y_hat)

        # Gradient
        l.backward()

        # Update
        optimiser.step()
        optimiser.zero_grad()

        # Log
        if epoch % 1 == 0:
            print(f"epoch {epoch+1}: w = {w.item():.3f}, loss = {l:.8f}")
    
    print(f"Prediction after training: f(5) = {(w*5):.3f}")


def torch():
    import torch as th
    import torch.nn as nn

    X = th.tensor([1,2,3,4], dtype=th.float32).view(4,1)
    Y = th.tensor([1,2,3,4], dtype=th.float32).view(4,1)

    X_test = th.tensor([5], dtype=th.float32)
    
    n_samples, n_features = X.shape

    input_size = n_features
    output_size = n_features

    model = nn.Linear(input_size, output_size, dtype=th.float32)
    
    # Training
    learning_rate = 0.05
    epochs = 10

    loss = nn.MSELoss()
    optimiser = th.optim.SGD(model.parameters(), lr=learning_rate)

    print(f"Prediction before training: f(5) = {model(X_test).item():.3f}")
    for epoch in range(epochs):
        
        # Forward
        y_hat = model(X)
        l = loss(Y, y_hat)

        # Gradient
        l.backward()

        # Update
        optimiser.step()
        optimiser.zero_grad()

        # Log
        if epoch % 1 == 0:
            [w, b] = model.parameters()
            print(f"epoch {epoch+1}: w = {w[0][0]}, loss = {l:.8f}")

def torch_custom():
    import torch as th
    import torch.nn as nn

    X = th.tensor([1,2,3,4], dtype=th.float32).view(4,1)
    Y = th.tensor([1,2,3,4], dtype=th.float32).view(4,1)

    X_test = th.tensor([5], dtype=th.float32)
    
    n_samples, n_features = X.shape

    input_size = n_features
    output_size = n_features

    #model = nn.Linear(input_size, output_size, dtype=th.float32)
    
    class LinearRegression(nn.Module):
        
        def __init__(self, input_dim, output_dim) -> None:
            super(LinearRegression, self).__init__()

            self.lin = nn.Linear(input_dim, output_dim)
        
        def forward(self, x):
            return self.lin(x)

    model = LinearRegression(input_size, output_size)

    # Training
    learning_rate = 0.05
    epochs = 50

    loss = nn.MSELoss()
    optimiser = th.optim.SGD(model.parameters(), lr=learning_rate)

    print(f"Prediction before training: f(5) = {model(X_test).item():.3f}")
    for epoch in range(epochs):
        
        # Forward
        y_hat = model(X)
        l = loss(Y, y_hat)

        # Gradient
        l.backward()

        # Update
        optimiser.step()
        optimiser.zero_grad()

        # Log
        if epoch % 1 == 0:
            [w, b] = model.parameters()
            print(f"epoch {epoch+1}: w = {w[0][0]}, loss = {l:.8f}")

if __name__ == '__main__':
    torch_custom()