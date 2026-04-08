
from activation_functions import tanh

def forward(X, params):
    "Neural network forward pass"
    activation_fn = tanh
    cache = {"A0": X}

    A = X
    L = len([k for k in params if k.startswith("W")]) 

    for l in range(1, L + 1):
        W, b = params[f"W{l}"], params[f"b{l}"]
        Z = A.dot(W.T) + b
        if l < L:  
            A = activation_fn(Z)
        else:      
            A = Z
        cache[f"Z{l}"], cache[f"A{l}"] = Z, A

    y_pred = A.ravel()
    return y_pred, cache