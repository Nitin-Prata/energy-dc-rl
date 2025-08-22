# Add project root to sys.path before any other imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from envs.data_center_env import DataCenterEnv


def make_env(seed=0):
    """Factory for environment creation."""
    def _init():
        env = DataCenterEnv()
        # Seeding in Gymnasium style
        try:
            env.reset(seed=seed)
        except TypeError:
            pass  # If DataCenterEnv doesn't accept seed in reset
        return Monitor(env)
    return _init


def train_and_evaluate(timesteps=10000, eval_episodes=5, seed=42):
    import pandas as pd
    # Vectorized environment with 4 parallel instances
    vec_env = DummyVecEnv([make_env(seed + i) for i in range(4)])

    # Initialize PPO model with tuned hyperparameters
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        seed=seed,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gae_lambda=0.98,
        gamma=0.99,
        ent_coef=0.01,
        clip_range=0.2
    )

    # Train the model longer for better learning
    model.learn(total_timesteps=50000)

    # Save the model
    model_path = os.path.join("results", "ppo_model")
    os.makedirs("results", exist_ok=True)
    model.save(model_path)

    # Evaluation and save per-episode metrics
    eval_env = make_env(seed=seed)()
    episodes = 20
    rewards, energies, costs, carbons, times = [], [], [], [], []
    for ep in range(episodes):
        obs, info = eval_env.reset()
        done = False
        ep_reward, ep_energy, ep_cost, ep_carbon, ep_time = 0, 0, 0, 0, 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            next_obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            ep_reward += reward
            ep_energy += info.get("step_energy", 0)
            ep_cost += info.get("step_cost", 0)
            ep_carbon += info.get("step_carbon", 0)
            ep_time = info.get("task_completion_time", ep_time)
            obs = next_obs
        rewards.append(ep_reward)
        energies.append(ep_energy)
        costs.append(ep_cost)
        carbons.append(ep_carbon)
        times.append(ep_time)
    df = pd.DataFrame({
        "episode": np.arange(1, episodes+1),
        "reward": rewards,
        "energy": energies,
        "total_cost": costs,
        "total_carbon": carbons,
        "completion_time": times
    })
    df.to_csv(os.path.join("results", "metrics.csv"), index=False)
    print(f"Training finished. Logs saved to results/")

    # ...existing code...
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=10000, help="Number of training timesteps")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train_and_evaluate(
        timesteps=args.timesteps,
        eval_episodes=args.eval_episodes,
        seed=args.seed
    )
