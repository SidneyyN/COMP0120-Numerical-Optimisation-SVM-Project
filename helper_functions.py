import numpy as np
import matplotlib.pyplot as plt

def generate_dataset(n_pos = 50, n_neg = 50, sigma = 0.8, seed = 0):
    rng = np.random.default_rng(seed)

    X_pos = rng.standard_normal((n_pos, 2)) * sigma + np.array([2.0, 2.0])
    X_neg = rng.standard_normal((n_pos, 2)) * sigma + np.array([-2.0, -2.0])

    X = np.vstack([X_pos, X_neg])
    y = np.hstack([np.ones(n_pos), -np.ones(n_neg)])

    return X, y

def build_Q(X, y):
    K = X @ X.T
    Q = np.outer(y, y) * K
    return Q 

def dual_objective(alpha, Q):
    return 0.5 * alpha @ (Q @ alpha) - np.sum(alpha)

def dual_gradient(alpha, Q):
    return Q @ alpha - np.ones_like(alpha)

def project_to_hyperplane(z, y):
    return z - y * ((y @ z) / (y @ y))

def project_to_box(z, C):
    return np.clip(z, 0.0, C)

# The projection onto the feasible set was implemented by alternating projection 
# onto the equality-constraint hyperplane and the box constraints.
def project_to_feasible_set(z, y, C, n_cycles=10):
    alpha = z.copy()
    for _ in range(n_cycles):
        alpha = project_to_hyperplane(alpha, y)
        alpha = project_to_box(alpha, C)
    return alpha

def plot_classifier(X, y, alpha, w, b, C, tol=1e-6):
    pos = y == 1
    neg = y == -1
    sv = alpha > tol

    plt.figure(figsize=(7, 6))
    plt.scatter(X[pos, 0], X[pos, 1], label="+1 class")
    plt.scatter(X[neg, 0], X[neg, 1], label="-1 class")
    plt.scatter(X[sv, 0], X[sv, 1], s=120, facecolors='none', edgecolors='k', label="support vectors")

    x1_vals = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200)

    if abs(w[1]) > 1e-12:
        # Decision boundary
        x2_vals = -(w[0] * x1_vals + b) / w[1]
        plt.plot(x1_vals, x2_vals, label="decision boundary")

        # Margins
        x2_margin_pos = -(w[0] * x1_vals + b - 1) / w[1]
        x2_margin_neg = -(w[0] * x1_vals + b + 1) / w[1]
        plt.plot(x1_vals, x2_margin_pos, linestyle="--")
        plt.plot(x1_vals, x2_margin_neg, linestyle="--")

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Linear SVM solved by projected gradient descent")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_objective_history(history):
    plt.figure(figsize=(7, 5))
    plt.plot(history["objective"])
    plt.xlabel("Iteration")
    plt.ylabel("Dual objective value")
    plt.title("Objective history")
    plt.grid(True)
    plt.show()