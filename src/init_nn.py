import numpy as np

def init_network(layer_dims, activation="tanh"):
    "Initialize the neural network parameters and architecture"
    params = {}
    for i in range(1, len(layer_dims)):
        input_dim = layer_dims[i - 1]
        output_dim = layer_dims[i]
        params[f"W{i}"] = np.random.randn(output_dim, input_dim) * np.sqrt(1.0 / input_dim)
        params[f"b{i}"] = np.zeros(output_dim)
    params["activation"] = activation
    return params