import numpy as np
import matplotlib.pyplot as py
from helper_functions import *
import time 

X, y = generate_dataset(n_pos=50, n_neg=50, sigma=0.8, seed=0)

start_Q = time.perf_counter()
Q = build_Q(X, y)
end_Q = time.perf_counter()
print("Q build time (s):", end_Q - start_Q)

C = 1.0
L = estimate_lipschitz_constant(Q)
eta = 1.0 / L

start = time.perf_counter()
alpha, history = projected_gradient_svm(
    Q, y, C=C, eta=eta, tol=1e-6, max_iter=10000, proj_cycles=10
)
end = time.perf_counter()

solve_time = end - start
print("Solve time (s):", solve_time)
print("Average time per iteration (s):", solve_time / len(history["objective"]))

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

mask_margin = (alpha > 1e-4) & (alpha < 1.0 - 1e-4)
vals_margin = y[mask_margin] * (X[mask_margin] @ w + b)

print("Recovered b:", b)
print("Margin values:", vals_margin)

print("Memory for X (bytes):", X.nbytes)
print("Memory for y (bytes):", y.nbytes)
print("Memory for Q (bytes):", Q.nbytes)
print("Memory for alpha (bytes):", alpha.nbytes)