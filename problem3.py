# run_rosenbrock.py
"""
FM216 — QUESTION 3: ROSENBROCK FUNCTION OPTIMIZATION

Improved version:
- Distinguishable line styles so GDAO is visible even if overlapping
- Stable plotting to avoid vertical jumps
- Clean summary table
- Saves plot to plots/ folder
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time

from gdao import GDAO
from baseline_optimizers import Adam, SGDM, RMSprop

# ensure output folder exists
os.makedirs("plots", exist_ok=True)

# ---------------------------------------------------------
# ROSENBROCK FUNCTION
# ---------------------------------------------------------
def f_rosen(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def grad_rosen(x):
    dx = -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2)
    dy = 200*(x[1] - x[0]**2)
    return np.array([dx, dy])

# ---------------------------------------------------------
# Generic optimizer runner
# ---------------------------------------------------------
def run_optimizer(opt, x0, tol=1e-4, max_iters=50000):
    x = x0.copy()
    losses = []

    start = time.time()

    for it in range(1, max_iters + 1):
        x, _ = opt.step(x, grad_rosen)
        loss = f_rosen(x)
        losses.append(loss)

        if loss < tol:
            duration = time.time() - start
            return it, losses, x, loss, duration

    duration = time.time() - start
    return max_iters, losses, x, f_rosen(x), duration

# ---------------------------------------------------------
# MAIN EXPERIMENT
# ---------------------------------------------------------
def run():
    print("\n===============================")
    print("   ROSENBROCK BENCHMARK (Q3)")
    print("===============================")

    x0 = np.array([-1.2, 1.0])

    optimizers = {
        "GDAO": GDAO(lr=0.001, gamma=0.7),
        "Adam": Adam(lr=0.001),
        "SGDM": SGDM(lr=0.001, beta=0.9),
        "RMSprop": RMSprop(lr=0.001)
    }

    results = {}

    print("Start point = (-1.2, 1.0)")
    print("Stopping condition: f(x, y) < 1e-4\n")

    for name, opt in optimizers.items():
        print(f"▶ Running {name}...")
        its, losses, x_final, final_loss, time_taken = run_optimizer(opt, x0)

        results[name] = {
            "iterations": its,
            "losses": losses,
            "x_final": x_final,
            "final_loss": final_loss,
            "time": time_taken
        }

        print(f"{name} finished:")
        print(f"  ➤ Iterations : {its}")
        print(f"  ➤ Final x    : {x_final}")
        print(f"  ➤ Final loss : {final_loss:.6e}")
        print(f"  ➤ Time taken : {time_taken:.4f} seconds\n")

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
    # PLOTTING — Distinguishable Lines
    # ---------------------------------------------------------
    style = {
        "GDAO":    {"color": "blue",   "linewidth": 2.5, "linestyle": "-"},
        "Adam":    {"color": "orange", "linewidth": 1.8, "linestyle": "--"},
        "SGDM":    {"color": "green",  "linewidth": 1.8, "linestyle": "-."},
        "RMSprop": {"color": "red",    "linewidth": 1.8, "linestyle": ":"},
    }

    plt.figure(figsize=(9, 5))
    for name, r in results.items():
        # Smooth curve (avoids spikes)
        y = np.array(r["losses"])
        y = np.maximum(y, 1e-12)

        plt.plot(
            y,
            label=name,
            color=style[name]["color"],
            linestyle=style[name]["linestyle"],
            linewidth=style[name]["linewidth"],
            alpha=0.9
        )

    plt.yscale("log")
    plt.xlabel("Iterations", fontsize=12)
    plt.ylabel("Loss (log scale)", fontsize=12)
    plt.title("Rosenbrock Function — GDAO vs Baselines", fontsize=14)
    plt.grid(True, linestyle="--", linewidth=0.6)
    plt.legend()

    plt.tight_layout()
    plt.savefig("plots/rosenbrock_comparison.png")
    plt.close()

    print("\nPlot saved → plots/rosenbrock_comparison.png")
    print("Done!\n")


if __name__ == "__main__":
    run()
