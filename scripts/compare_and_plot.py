# scripts/compare_and_plot.py
"""
Load baseline results (results/baseline/...), RL results (runs/metrics.csv) and produce
comparison plots and a small CSV summary (improvement %).
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = "results/baseline"
RL_RUN_DIR = "runs"

def load_baseline_agg():
    files = sorted(glob.glob(os.path.join(BASE_DIR, "baseline_ep*.csv")))
    energies = []
    times = []
    for f in files:
        df = pd.read_csv(f)
        energies.append(df["energy"].sum())
        times.append(df["avg_completion_time"].mean())
    return np.array(energies), np.array(times)

def load_rl_metrics():
    path = os.path.join(RL_RUN_DIR, "metrics.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return df

def main():
    be, bt = load_baseline_agg()
    rl = load_rl_metrics()

    os.makedirs("results/plots", exist_ok=True)

    # Baseline bars
    plt.figure(figsize=(8,5))
    plt.bar(range(len(be)), be)
    plt.xlabel("Baseline episode")
    plt.ylabel("Total energy")
    plt.title("Baseline energy per episode")
    plt.savefig("results/plots/baseline_energy_bar.png")
    plt.close()

    if rl is not None:
        plt.figure(figsize=(8,5))
        plt.plot(rl["episode"], rl["energy"], label="RL energy")
        plt.hlines(be.mean(), 1, max(rl["episode"]), colors="red", linestyles="dashed", label="baseline mean")
        plt.xlabel("Episode")
        plt.ylabel("Energy")
        plt.legend()
        plt.title("RL vs Baseline energy")
        plt.savefig("results/plots/rl_vs_baseline_energy.png")
        plt.close()

        # summary
        summary = {
            "baseline_energy_mean": float(be.mean()),
            "rl_energy_mean": float(rl["energy"].mean()),
            "energy_reduction_pct": float(100.0 * (be.mean() - rl["energy"].mean()) / be.mean())
        }
        pd.DataFrame([summary]).to_csv("results/summary.csv", index=False)
        print("Saved results/summary.csv")

    print("Plots saved to results/plots")

if __name__ == "__main__":
    main()
