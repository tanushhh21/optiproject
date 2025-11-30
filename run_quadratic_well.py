"""
FM216 — QUESTION 1: WELL-CONDITIONED QUADRATIC
Generates the convergence plot:
plots/quadratic_well_comparison.png
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt

from gdao import GDAO
from baseline_optimizers import Adam, SGDM, RMSprop

# Create plots directory
os.makedirs("plots", exist_ok=True)

# ------------------------------------------------------
# Create well-conditioned quadratic (κ ≤ 10)
# ------------------------------------------------------
def create_well(n=10, seed=42):
    np.random.seed(seed)
    eigen = np.linspace(1, 10, n)  # κ = 10
    Q, _ = np.linalg.qr(np.random.randn(n, n))
    Q = Q @ np.diag(eigen) @ Q.T

    def f(x):
        return 0.5 * (x @ Q @ x)

    def grad(x):
        return Q @ x

    return f, grad

# ------------------------------------------------------
# Run any optimizer until f(x) < tol
# ------------------------------------------------------
def run_optimizer(opt, x0, f, grad, tol=1e-6, max_iters=20000):
    x = x0.copy()
    losses = []
    start = time.time()

    for it in range(1, max_iters + 1):
        x, _ = opt.step(x, grad)
        L = f(x)
        losses.append(L)

        if L < tol:
            duration = time.time() - start
            return it, losses, x, L, duration

    duration = time.time() - start
    return max_iters, losses, x, f(x), duration

# ------------------------------------------------------
# MAIN EXPERIMENT
# ------------------------------------------------------
def run():
    print("\n===============================")
    print(" WELL-CONDITIONED QUADRATIC (Q1)")
    print("===============================\n")

    dim = 10
    f, grad = create_well(dim)
    x0 = np.ones(dim)

    print(f"Dimension: {dim}")
    print("Condition number: 10")
    print("Goal: f(x) < 1e-6\n")

    optimizers = {
        "GDAO":   GDAO(lr=0.01, gamma=0.5),
        "Adam":   Adam(lr=0.001),
        "SGDM":   SGDM(lr=0.01, beta=0.9),
        "RMSprop": RMSprop(lr=0.001)
    }

    results = {}

    for name, opt in optimizers.items():
        print(f"▶ Running {name}...")
        its, losses, x_final, final_loss, time_taken = run_optimizer(opt, x0, f, grad)

        results[name] = {
            "iterations": its,
            "losses": losses,
            "final_loss": final_loss,
            "time": time_taken
        }

        print(f"{name} finished:")
        print(f"  ➤ Iterations : {its}")
        print(f"  ➤ Final loss : {final_loss:.6e}")
        print(f"  ➤ Time taken : {time_taken:.4f} sec\n")

    # ------------------------------------------------------
    # SUMMARY TABLE
    # ------------------------------------------------------
    print("==========================================")
    print("           FINAL COMPARISON TABLE         ")
    print("==========================================")
    print(f"{'Optimizer':<10} | {'Iters':<10} | {'Final Loss':<12} | Time (s)")
    print("-"*55)
    for name, r in results.items():
        print(f"{name:<10} | {r['iterations']:<10} | {r['final_loss']:<12.4e} | {r['time']:.4f}")
    print("-"*55)

    # ------------------------------------------------------
    # PLOT CURVES
    # ------------------------------------------------------
    plt.figure(figsize=(8,5))
    for name, r in results.items():
        plt.plot(r["losses"], label=name)

    plt.yscale("log")
    plt.xlabel("Iterations")
    plt.ylabel("Loss (log scale)")
    plt.title("Well-Conditioned Quadratic (κ = 10)")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/quadratic_well_comparison.png")
    plt.close()

    print("\nPlot saved → plots/quadratic_well_comparison.png")
    print("Done!\n")


if __name__ == "__main__":
    run()
