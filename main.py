import numpy as np
import matplotlib.pyplot as py
from helper_functions import *

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


X, y = generate_dataset(n_pos=50, n_neg=50, sigma=0.8, seed=0)
Q = build_Q(X, y)

C = 1.0
L = estimate_lipschitz_constant(Q)
eta = 1.0 / L

alpha, history = projected_gradient_svm(
    Q, y, C=C, eta=eta, tol=1e-6, max_iter=10000, proj_cycles=10
)

w, b = recover_primal_variables(X, y, alpha, C=C)

print("Final objective:", dual_objective(alpha, Q))
print("Iterations:", len(history["objective"]))
print("Equality violation:", abs(y @ alpha))
print("Number of support vectors:", np.sum(alpha > 1e-6))

plot_classifier(X, y, alpha, w, b, C)
plot_objective_history(history)

y_pred = np.sign(X @ w + b)
accuracy = np.mean(y_pred == y)
print("Training accuracy:", accuracy)

mask = (alpha > 1e-4) & (alpha < C - 1e-4)
print("Margin support vectors:", np.sum(mask))

print("Final step norm:", history["step_norm"][-1])


# mask_margin = (alpha > 1e-4) & (alpha < C - 1e-4)

# vals_all = y * (X @ w + b)
# vals_margin = vals_all[mask_margin]

# print("All values min/max:", np.min(vals_all), np.max(vals_all))
# print("Margin values:", vals_margin)
# print("Alpha values on margin mask:", alpha[mask_margin])
# print("y - Xw on margin mask:", y[mask_margin] - X[mask_margin] @ w)
# print("Current b:", b)
mask_margin = (alpha > 1e-4) & (alpha < 1.0 - 1e-4)
vals_margin = y[mask_margin] * (X[mask_margin] @ w + b)

print("Recovered b:", b)
print("Margin values:", vals_margin)