import numpy as np

np.random.seed(42)

def generate_data(N=200):
    "Generate synthetic data for regression"
    x = np.random.uniform(-np.pi, np.pi, size=N)
    X = x.reshape(-1, 1)
    y = (np.tanh(x**2 + x - 1))**5 + np.random.normal(scale=0.08, size=N)
    return X, y