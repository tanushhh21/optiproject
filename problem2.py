# run_quadratic_ill.py
"""
FM216 — QUESTION 2: ILL-CONDITIONED QUADRATIC

Upgraded version:
- Clearer, distinguishable plot (GDAO visible)
- Smooth curves (avoid spikes when loss becomes small)
- Clean summary table
- Saves plot to plots/quadratic_ill_comparison.png
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt

from gdao import GDAO
from baseline_optimizers import Adam, SGDM, RMSprop

os.makedirs("plots", exist_ok=True)

# ---------------------------------------------------------
# Create ill-conditioned quadratic
# ---------------------------------------------------------
def create_ill(n=10, cond=500):
    eigen = np.linspace(1, cond, n)
    Q, _ = np.linalg.qr(np.random.randn(n, n))
    Q = Q @ np.diag(eigen) @ Q.T

    def f(x):
        return 0.5 * (x @ Q @ x)

    def grad(x):
        return Q @ x

    return f, grad

# ---------------------------------------------------------
# Run optimizer
# ---------------------------------------------------------
def run_optimizer(opt, x0, f, grad, max_iters=30000, tol=1e-6):
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

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def run():
    print("\n===============================")
    print(" ILL-CONDITIONED QUADRATIC (Q2)")
    print("===============================\n")

    dim = 10
    cond = 500
    x0 = np.ones(dim)

    f, grad = create_ill(dim, cond)

    print(f"Dimension: {dim}")
    print(f"Condition number: {cond}")
    print("Goal: f(x) < 1e-6\n")

    # Optimizers
    optimizers = {
        "GDAO": GDAO(lr=0.001, gamma=0.5),
        "Adam": Adam(lr=0.001),
        "SGDM": SGDM(lr=0.01, beta=0.9),
        "RMSprop": RMSprop(lr=0.001)
    }

    results = {}

    # ---------------------------------------------------------
    # RUN OPTIMIZERS
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # SUMMARY TABLE
    # ---------------------------------------------------------
    print("==========================================")
    print("           FINAL COMPARISON TABLE         ")
    print("==========================================")
    print(f"{'Optimizer':<10} | {'Iters':<10} | {'Final Loss':<12} | Time (s)")
    print("-"*55)
    for name, r in results.items():
        print(f"{name:<10} | {r['iterations']:<10} | {r['final_loss']:<12.4e} | {r['time']:.4f}")
    print("-"*55)

    # ---------------------------------------------------------
    # PLOT — Distinguishable curves
    # ---------------------------------------------------------
    style = {
        "GDAO":    {"color": "blue",   "linewidth": 2.5, "linestyle": "-"},
        "Adam":    {"color": "orange", "linewidth": 1.8, "linestyle": "--"},
        "SGDM":    {"color": "green",  "linewidth": 1.8, "linestyle": "-."},
        "RMSprop": {"color": "red",    "linewidth": 1.8, "linestyle": ":"},
    }

    plt.figure(figsize=(8,5))

    for name, r in results.items():
        y = np.array(r["losses"])
        y = np.maximum(y, 1e-12)  # avoid log plot issues

        plt.plot(
            y,
            label=name,
            color=style[name]["color"],
            linestyle=style[name]["linestyle"],
            linewidth=style[name]["linewidth"]
        )

    plt.yscale("log")
    plt.xlabel("Iterations")
    plt.ylabel("Loss (log scale)")
    plt.title(f"Ill-Conditioned Quadratic (κ = {cond})")
    plt.grid(True, linestyle="--", linewidth=0.6)
    plt.legend()

    plt.tight_layout()
    plt.savefig("plots/quadratic_ill_comparison.png")
    plt.close()

    print("\nPlot saved → plots/quadratic_ill_comparison.png")
    print("Done!\n")


if __name__ == "__main__":
    run()
