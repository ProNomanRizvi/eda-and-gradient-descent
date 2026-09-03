# Gradient Descent from Scratch

## Industry Context

A common way to prove real understanding of ML fundamentals  rather than
treating `model.fit()` as a black box  is to implement linear regression's
training loop by hand: define a loss function, derive its gradient, and
iteratively update parameters, then verify the result against a trusted
library implementation (`sklearn`). This is a standard beginner-to-intermediate
"ML from scratch" portfolio project that demonstrates foundational
understanding rather than just library usage. This project pulls together
matrix operations, calculus-based gradients, and visualization into one
working implementation.

## Core Concepts Practiced

**Multi-feature linear model**
```
y_pred = X @ weights + bias
```

`X` is a feature matrix (rows = samples, columns = features). Matrix
multiplication computes predictions for every sample at once, rather than
looping row by row.

**Loss function — Mean Squared Error (MSE)**
```
MSE = (1/n) * sum((y_pred - y_actual)^2)
```

Squaring removes the sign of the error and penalizes bigger mistakes more
heavily than small ones.

**Gradient of MSE**
```
dW = (2/n) * X.T @ (y_pred - y_actual)
db = (2/n) * sum(y_pred - y_actual)
```

The partial derivative of the loss with respect to each parameter,
computed for every weight simultaneously via matrix multiplication.

**Parameter update rule**
```
weights -= learning_rate * dW
bias -= learning_rate * db
```

Move against the gradient (direction of steepest increase) by a step
sized by the learning rate, moving in the *opposite* direction decreases
the loss.

**Convergence isn't just "does the loss curve look flat"**
A loss curve can look fully converged (visually flat) while individual
parameters are still slowly drifting toward their optimum; different
parameters can converge at different rates. In this run, weights matched
sklearn's closed-form solution almost immediately, but bias was still
measurably off (~0.50 difference) at 1000 iterations. Increasing to 3000
iterations let bias finish converging (~0.005 difference from sklearn)
without changing anything else in the model. Lesson: track the actual
parameter values, not just the loss curve, when deciding whether training
is done.

**Verifying against a trusted implementation**
Comparing the from-scratch result to `sklearn.linear_model.LinearRegression`
(which solves the same problem via a different optimization approach, not
gradient descent) is the strongest available check that the hand-derived
math and code are correct; if both methods converge to the same answer
independently, the implementation is almost certainly right.

## Task

`gradient_descent_from_scratch.py`
* Synthetic dataset: 200 samples, 3 features, generated from a known ground
  truth (`true_weights = [3.5, -2.0, 1.2]`, `true_bias = 5.0`) plus Gaussian
  noise so the "correct answer" is known in advance.
* Full manual training loop (no library shortcuts): predictions, MSE loss
  (tracked every iteration for the convergence plot), gradients, parameter
  updates. 3000 iterations, `learning_rate=0.01`.
* Learned weights and bias compared directly against both the true
  generating parameters and sklearn's `LinearRegression`fit on the same
  data.
* Convergence plot (OO-style Matplotlib): MSE loss vs. iteration, showing
  the steep initial drop and long flat tail typical of gradient descent on
  a well-conditioned loss surface.

## Output & Findings

| Item | Result |
|---|---|
| Learned weights | `[3.5039, -2.0295, 1.1924]` vs. true `[3.5, -2.0, 1.2]` |
| Learned bias | `5.1105` vs. true `5.0` |
| Difference from sklearn (weights) | ~0.0003 |
| Difference from sklearn (bias) | ~0.005 |
| `convergence_plot.png` | MSE loss vs. iteration: steep initial drop, long flat tail |

The from-scratch implementation converges to within a fraction of a
percent of sklearn's closed-form solution, confirming the manual gradient
derivation and update loop are mathematically correct.

## Notes

This project was originally built as part of the ml-engineer-journey
learning log (Phase 5) and has since graduated into this standalone
portfolio repository.