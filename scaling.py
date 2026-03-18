import time 
from helper_functions import *

sizes = [40, 80, 120, 160]
results = []

for m_each_class in [20, 40, 60, 80]:
    X, y = generate_dataset(n_pos=m_each_class, n_neg=m_each_class, sigma=0.8, seed=0)
    m = len(y)

    t0 = time.perf_counter()
    Q = build_Q(X, y)
    t1 = time.perf_counter()

    L = estimate_lipschitz_constant(Q)
    eta = 1.0 / L

    t2 = time.perf_counter()
    alpha, history = projected_gradient_svm(
        Q, y, C=1.0, eta=eta, tol=1e-6, max_iter=10000, proj_cycles=10
    )
    t3 = time.perf_counter()

    results.append({
        "m": m,
        "Q_build_time": t1 - t0,
        "solve_time": t3 - t2,
        "iterations": len(history["objective"]),
        "time_per_iter": (t3 - t2) / len(history["objective"]),
        "Q_MB": Q.nbytes / (1024 ** 2),
        "X_MB": X.nbytes / (1024 ** 2)
    })

for r in results:
    print(r)