from activation_functions import dtanh
import numpy as np

def backward(y_pred, y_true, params, cache):
    "Neural network backward pass to compute gradients"
    L = len([k for k in params if k.startswith("W")])
    grads = {}
    N = y_true.shape[0]

    dA = (2.0 / N) * (y_pred - y_true).reshape(N, 1)

    for l in range(L, 0, -1):
        A_prev = cache[f"A{l-1}"]
        Z = cache[f"Z{l}"]
        W = params[f"W{l}"]

        if l == L:
            dZ = dA  
        else:
            act_fn = dtanh
            dZ = dA * act_fn(cache[f"A{l}"])

        grads[f"dW{l}"] = dZ.T.dot(A_prev)
        grads[f"db{l}"] = np.mean(dZ, axis=0)
        dA = dZ.dot(W)

    return grads
