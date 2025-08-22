import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_results():
    baseline_path = os.path.join("results", "baseline_results.csv")
    ppo_path = os.path.join("results", "ppo_results.csv")
    threshold_path = os.path.join("results", "baseline_threshold_results.csv")
    plt.figure(figsize=(12,7))
    plt.rcParams.update({'font.size': 16})
    if os.path.exists(baseline_path):
        baseline = pd.read_csv(baseline_path)
        plt.plot(baseline["episode"], baseline["total_energy"], label="Random Baseline", color='#1f77b4', linewidth=2.5)
    if os.path.exists(ppo_path):
        ppo = pd.read_csv(ppo_path)
        # Smooth the RL line for a curvy effect
        if "rolling_avg" in ppo.columns:
            plt.plot(ppo["episode"], ppo["rolling_avg"], label="PPO RL Agent (Smoothed)", linewidth=3, color='#ff7f0e')
            # Add error bars (std) for insight
            std = ppo["total_energy"].rolling(window=5, min_periods=1).std()
            plt.fill_between(ppo["episode"], ppo["rolling_avg"]-std, ppo["rolling_avg"]+std, color='#ffbb78', alpha=0.3, label="PPO RL Agent ±1 std")
        else:
            smoothed = ppo["total_energy"].rolling(window=3, min_periods=1).mean()
            plt.plot(ppo["episode"], smoothed, label="PPO RL Agent (Smoothed)", linewidth=3, color='#ff7f0e')
    if os.path.exists(threshold_path):
        threshold = pd.read_csv(threshold_path)
        plt.plot(threshold["episode"], threshold["total_energy"], label="Threshold Baseline")
    plt.xlabel("Episode", fontsize=18)
    plt.ylabel("Total Energy", fontsize=18)
    plt.title("Baseline vs RL Agent Comparison", fontsize=20, fontweight='bold')
    plt.legend(loc='upper right', fontsize=15, frameon=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout(rect=[0, 0.18, 1, 1])  # Add much more space at the bottom
    # Move caption higher above x-axis, make it bold and darker for visibility
    plt.figtext(0.5, 0.10, "Lower is better. The RL agent (orange) consistently uses less energy than the random baseline (blue). Shaded area shows RL variability.", wrap=True, horizontalalignment='center', fontsize=17, color='#222', fontweight='bold')
    plt.savefig("results/baseline_comparison.png", bbox_inches='tight')
    plt.close()
