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

def projected_gradient_svm(Q, y, C, eta, tol=1e-6, max_iter=5000, proj_cycles=10):
    m = len(y)
    alpha = np.zeros(m)
    history = {
        "objective": [],
        "step_norm": [],
        "feas_eq": [],
        "feas_box": []
    }

    for k in range(max_iter):
        grad = dual_gradient(alpha, Q)

        # Gradient step
        z = alpha - eta * grad

        # Projection
        alpha_new = project_to_feasible_set(z, y, C, n_cycles=proj_cycles)

        # Diagnostics
        obj = dual_objective(alpha_new, Q)
        step_norm = np.linalg.norm(alpha_new - alpha)
        eq_violation = abs(y @ alpha_new)
        box_violation = max(np.max(-alpha_new), np.max(alpha_new - C))

        history["objective"].append(obj)
        history["step_norm"].append(step_norm)
        history["feas_eq"].append(eq_violation)
        history["feas_box"].append(max(0.0, box_violation))

        # Stopping criterion
        if step_norm < tol:
            alpha = alpha_new
            break

        alpha = alpha_new

    return alpha, history

def estimate_lipschitz_constant(Q):
    return np.linalg.norm(Q, 2)

def recover_primal_variables(X, y, alpha, C, tol=1e-4):
    w = ((alpha * y)[:, None] * X).sum(axis=0)

    mask_margin = (alpha > tol) & (alpha < C - tol)

    if np.any(mask_margin):
        b_vals = y[mask_margin] - X[mask_margin] @ w
        b = np.mean(b_vals)
    else:
        mask_sv = alpha > tol
        if np.any(mask_sv):
            b_vals = y[mask_sv] - X[mask_sv] @ w
            b = np.mean(b_vals)
        else:
            b = 0.0

    return w, b

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