from init_nn import init_network
from forward_pass import forward
from backpropagation import backward

def train(X, y, layer_dims, lr=0.01, iters=5000, activation="tanh"):
    params = init_network(layer_dims, activation)

    for _ in range(1, iters + 1):
        y_pred, cache = forward(X, params)
        grads = backward(y_pred, y, params, cache)

        L = len([k for k in params if k.startswith("W")])
        for l in range(1, L + 1):
            #Update parameters using gradient descent
            params[f"W{l}"] -= lr * grads[f"dW{l}"]
            params[f"b{l}"] -= lr * grads[f"db{l}"]

    return params