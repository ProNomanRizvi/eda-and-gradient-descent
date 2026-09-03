import numpy as np
from sklearn.linear_model import LinearRegression

from gradient_descent_from_scratch import generate_dataset, train_gradient_descent


def test_weights_converge_close_to_true_weights():
    X, y, true_weights, true_bias = generate_dataset()
    weights, bias, _ = train_gradient_descent(X, y)

    assert np.allclose(weights, true_weights, atol=0.05)
    assert abs(bias - true_bias) < 0.2


def test_loss_decreases_monotonically_on_average():
    X, y, _, _ = generate_dataset()
    _, _, loss_history = train_gradient_descent(X, y)

    # Early loss should be much higher than final loss.
    assert loss_history[0] > loss_history[-1]

    # Loss in the second half of training should be stable,
    # not still dropping sharply.
    second_half = loss_history[len(loss_history) // 2:]
    assert max(second_half) - min(second_half) < 0.5


def test_matches_sklearn_within_tolerance():
    X, y, _, _ = generate_dataset()
    weights, bias, _ = train_gradient_descent(X, y)

    sk_model = LinearRegression()
    sk_model.fit(X, y)

    weight_diff = np.abs(weights - sk_model.coef_)
    bias_diff = abs(bias - sk_model.intercept_)

    assert np.all(weight_diff < 0.001)
    assert bias_diff < 0.01