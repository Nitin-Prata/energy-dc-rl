# Baseline runner for threshold autoscaler
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import csv
from envs.data_center_env import DataCenterEnv
from baselines.threshold_autoscaler import threshold_autoscaler

def run_baseline(num_episodes=5, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "baseline_threshold_results.csv")
    env = DataCenterEnv()
    results = []
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_energy = 0
        steps = 0
        current_active = 1
        max_servers = 5
        while not done:
            action = np.array([threshold_autoscaler(obs, current_active, max_servers)])
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_energy += abs(reward)  # or use info if available
            steps += 1
        results.append({
            "episode": episode + 1,
            "total_energy": total_energy,
            "steps": steps
        })
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "total_energy", "steps"])
        writer.writeheader()
        writer.writerows(results)
    print(f"[INFO] Threshold baseline results saved to {csv_path}")

if __name__ == "__main__":
    run_baseline()
