def main():
    from data_generation import generate_data
    from train import train
    from forward_pass import forward
    import matplotlib.pyplot as plt
    import numpy as np
    
    X, y = generate_data(N=200)

    #Plot the data
    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    ax.scatter(X.flatten(), y, s=10, label="data")
    ax.legend()
    ax.set_xlabel("Input Feature (x)")
    ax.set_ylabel("Target Value (y)")
    plt.tight_layout()

    plt.savefig("figures/data.png")

    #Init the network
    params = train(X, y, [1, 10, 10, 10, 10, 10, 1], lr=0.01, iters=20000, activation="tanh")
    
    #Test out the model
    y_pred, _ = forward(X, params)
    
    # Sort for plotting
    sort_idx = np.argsort(X.flatten())
    X_sorted = X[sort_idx]
    y_pred_sorted = y_pred[sort_idx]
    
    # Plot the regression line
    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    ax.scatter(X.flatten(), y, s=10, label="data")
    ax.plot(X_sorted.flatten(), y_pred_sorted.flatten(), color="red", label="model")
    ax.legend()
    ax.set_xlabel("Input Feature (x)")
    ax.set_ylabel("Target Value (y)")
    plt.tight_layout()

    plt.savefig("figures/model.png")

if __name__ == "__main__":
    main()