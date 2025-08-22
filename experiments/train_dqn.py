# experiments/train_dqn.py
"""
Train script for DQN agent on DataCenterEnv.
Saves:
 - checkpoint: runs/checkpoint_latest.pt
 - per-episode metrics: runs/metrics.csv
 - plots in runs/plots/
"""
import os
import math
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# ensure project root import works
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from envs.data_center_env import DataCenterEnv
from agents.dqn import DQNAgent

def epsilon_by_frame(frame_idx, eps_start=1.0, eps_final=0.05, eps_decay=40000):
    return eps_final + (eps_start - eps_final) * math.exp(-1. * frame_idx / eps_decay)

def train(seed=0, episodes=300, max_steps_per_ep=200, log_dir="runs"):
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, "plots"), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = DataCenterEnv(seed=seed)
    obs, _ = env.reset()
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0] if hasattr(env.action_space, 'shape') else env.action_space.n

    agent = DQNAgent(obs_dim, n_actions, lr=1e-4, batch_size=64,
                     buffer_capacity=200000, target_update_freq=1000, dueling=True)
    print(f"Using device: {device}")

    metrics = []
    frame_idx = 0
    start_time = time.time()

    for ep in range(episodes):
        reset_result = env.reset()
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            state, _ = reset_result
        else:
            state = reset_result

        ep_reward = 0.0
        ep_energy = 0.0
        ep_completion_time = 0.0

        done = False
        step = 0
        while not done and step < max_steps_per_ep:
            eps = epsilon_by_frame(frame_idx)
            action = agent.select_action(state, eps)
            step_result = env.step(action)
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_state, reward, done, info = step_result

            agent.push(state, action, reward, next_state, done)
            loss, _ = agent.update()

            ep_reward += reward
            ep_energy += info.get("step_energy", 0)
            ep_completion_time = info.get("task_completion_time", ep_completion_time)

            state = next_state
            step += 1
            frame_idx += 1

        metrics.append({
            "episode": ep+1,
            "reward": ep_reward,
            "energy": ep_energy,
            "completion_time": ep_completion_time,
            "total_cost": info.get("total_cost", 0),
            "total_carbon": info.get("total_carbon", 0),
            "steps": step
        })

        # Print progress
        if (ep + 1) % 10 == 0 or ep == 0:
            elapsed = time.time() - start_time
            print(f"Ep {ep+1}/{episodes}  reward={ep_reward:.2f} energy={ep_energy:.1f} comp_time={ep_completion_time:.2f} frames={frame_idx} elapsed={elapsed:.1f}s")

        # checkpoint
        if (ep + 1) % 50 == 0:
            torch.save(agent.q_net.state_dict(), os.path.join(log_dir, f"dqn_ep{ep+1}.pt"))

    # final save
    torch.save(agent.q_net.state_dict(), os.path.join(log_dir, "dqn_final.pt"))
    df = pd.DataFrame(metrics)
    df.to_csv(os.path.join(log_dir, "metrics.csv"), index=False)

    # quick plots
    plt.figure(); plt.plot(df['episode'], df['energy']); plt.title("Energy per episode"); plt.savefig(os.path.join(log_dir,"plots","energy_per_episode.png")); plt.close()
    plt.figure(); plt.plot(df['episode'], df['completion_time']); plt.title("Completion time per episode"); plt.savefig(os.path.join(log_dir,"plots","completion_per_episode.png")); plt.close()

    print(f"Training finished. Logs saved to {log_dir}")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=200)
    args = parser.parse_args()
    train(seed=args.seed, episodes=args.episodes)
