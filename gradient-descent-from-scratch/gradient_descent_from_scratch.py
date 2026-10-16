# gradient_descent_from_scratch.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def generate_dataset(n_samples=200, seed=42):
    """Generate a synthetic multi-feature regression dataset."""
    rng = np.random.RandomState(seed)
    X = rng.rand(n_samples, 3) * 10
    true_weights = np.array([3.5, -2.0, 1.2])
    true_bias = 5.0
    noise = rng.normal(0, 1, n_samples)
    y = X @ true_weights + true_bias + noise
    return X, y, true_weights, true_bias


def train_gradient_descent(X, y, learning_rate=0.01, n_iterations=3000):
    """
    Train multi-feature linear regression via batch gradient descent.

    Returns learned weights, bias, and the loss history.
    """
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0.0
    loss_history = []

    for _ in range(n_iterations):
        y_pred = X @ weights + bias
        error = y_pred - y

        mse = np.mean(error ** 2)
        loss_history.append(mse)

        dW = (2 / n_samples) * (X.T @ error)
        db = (2 / n_samples) * np.sum(error)

        weights -= learning_rate * dW
        bias -= learning_rate * db

    return weights, bias, loss_history


def plot_convergence(loss_history, save_path="convergence_plot.png"):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(loss_history)
    ax.set_title("Gradient Descent Convergence")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("MSE Loss")
    ax.grid(True)
    fig.tight_layout()
    plt.savefig(save_path)
    plt.show()


def main():
    X, y, true_weights, true_bias = generate_dataset()
    weights, bias, loss_history = train_gradient_descent(X, y)

    print("=" * 60)
    print("FROM-SCRATCH GRADIENT DESCENT RESULTS")
    print("=" * 60)
    print("\nTrue weights:")
    print(true_weights)
    print("\nLearned weights:")
    print(weights)
    print("\nTrue bias:")
    print(true_bias)
    print("\nLearned bias:")
    print(bias)
    print("\nFinal MSE loss:")
    print(loss_history[-1])

    plot_convergence(loss_history)

    sk_model = LinearRegression()
    sk_model.fit(X, y)

    print("\n" + "=" * 60)
    print("SKLEARN LINEAR REGRESSION")
    print("=" * 60)
    print("\nSklearn weights:")
    print(sk_model.coef_)
    print("\nSklearn bias:")
    print(sk_model.intercept_)

    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print("\nDifference between scratch and sklearn weights:")
    print(weights - sk_model.coef_)
    print("\nDifference between scratch and sklearn bias:")
    print(bias - sk_model.intercept_)


if __name__ == "__main__":
    main()
