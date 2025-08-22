# Demo script: Run baseline and RL agent, plot results
import os
import subprocess
import matplotlib.pyplot as plt
import pandas as pd

def run_all():
    print("Running baseline...")
    subprocess.run(["python", "scripts/run_baseline.py"])
    print("Running threshold baseline...")
    subprocess.run(["python", "scripts/run_threshold_baseline.py"])
    print("Running PPO RL agent...")
    subprocess.run(["python", "experiments/train_ppo.py", "--timesteps", "20000", "--eval-episodes", "5"])
    print("Plotting results...")
    plot_results()

def plot_results():
    baseline = pd.read_csv("results/baseline_results.csv")
    threshold = pd.read_csv("results/baseline_threshold_results.csv")
    # Optionally, add RL results if saved as CSV
    plt.figure(figsize=(10,6))
    plt.plot(baseline["episode"], baseline["total_energy"], label="Random Baseline")
    plt.plot(threshold["episode"], threshold["total_energy"], label="Threshold Baseline")
    plt.xlabel("Episode")
    plt.ylabel("Total Energy")
    plt.title("Baseline Comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/baseline_comparison.png")
    plt.show()

if __name__ == "__main__":
    run_all()
