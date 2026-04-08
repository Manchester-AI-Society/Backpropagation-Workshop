import numpy as np

def tanh(x):
    "Activation function: hyperbolic tangent"
    return np.tanh(x)

def dtanh(x):
    "Derivative of the hyperbolic tangent"
    return 1 - x**2