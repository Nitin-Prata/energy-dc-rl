
# Add project root to sys.path before any other imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import matplotlib.pyplot as plt
from tqdm import trange
from envs.data_center_env import DataCenterEnv


def run_baseline(num_episodes=20, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "baseline_results.csv")

    env = DataCenterEnv()

    results = []
    for episode in trange(num_episodes, desc="Running Baseline"):
        obs = env.reset()
        done = False
        total_energy = 0
        steps = 0


        while not done:
            # Choose a random action for more variation
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_energy = info["total_energy"]
            steps += 1

        # Use the sum of completion times for this episode
        total_completion_time = env.total_completion_time
        results.append({
            "episode": episode + 1,
            "total_energy": total_energy,
            "total_completion_time": total_completion_time,
            "steps": steps
        })

    # Save CSV once
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "total_energy", "total_completion_time", "steps"])
        writer.writeheader()
        writer.writerows(results)

    print(f"[INFO] Baseline results saved to {csv_path}")

    # Plot from results in memory
    episodes = [r["episode"] for r in results]
    total_energy_vals = [r["total_energy"] for r in results]
    completion_time_vals = [r["total_completion_time"] for r in results]


    fig, ax1 = plt.subplots(figsize=(10, 5))
    color = 'tab:blue'
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Energy', color=color)
    ax1.plot(episodes, total_energy_vals, color=color, label='Total Energy')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)

    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('Total Completion Time', color=color)
    ax2.plot(episodes, completion_time_vals, color=color, label='Total Completion Time')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.suptitle('Baseline Performance')
    fig.tight_layout()
    plot_path = os.path.join(output_dir, "baseline_plot.png")
    plt.savefig(plot_path)
    plt.close(fig)

    print(f"[INFO] Plot saved to {plot_path}")
    print("[INFO] Baseline run complete.")


if __name__ == "__main__":
    run_baseline(num_episodes=5)
