# Benchmark script: RL agent vs baselines
import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.data_center_env import DataCenterEnv
from stable_baselines3 import PPO

def run_baseline(env, policy_fn, episodes=50):
    results = []
    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0.0
        total_energy = 0.0
        total_cost = 0.0
        total_carbon = 0.0
        while not done:
            action = policy_fn(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            total_energy += info["step_energy"]
            total_cost += info["step_cost"]
            total_carbon += info["step_carbon"]
            state = next_state
        results.append({"reward": total_reward, "energy": total_energy, "cost": total_cost, "carbon": total_carbon})
    return pd.DataFrame(results)

def threshold_policy(state):
    return 0.5  # Simple threshold

def random_policy(state):
    return np.random.uniform(0.0, 1.0)

def main():
    env = DataCenterEnv()
    print("Running threshold baseline...")
    df_thresh = run_baseline(env, threshold_policy)
    print("Running random baseline...")
    df_rand = run_baseline(env, random_policy)
    print("Running RL agent (PPO)...")
    model_path = os.path.join("results", "ppo_model.zip")
    if os.path.exists(model_path):
        model = PPO.load(model_path)
        results = []
        for ep in range(50):
            obs, _ = env.reset()
            done = False
            total_reward = 0.0
            total_energy = 0.0
            total_cost = 0.0
            total_carbon = 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
                total_energy += info["step_energy"]
                total_cost += info["step_cost"]
                total_carbon += info["step_carbon"]
                obs = next_obs
            results.append({"reward": total_reward, "energy": total_energy, "cost": total_cost, "carbon": total_carbon})
        df_rl = pd.DataFrame(results)
    else:
        print("No PPO model found. RL agent results will be empty.")
        df_rl = pd.DataFrame()
    # Save results
    df_thresh.to_csv("results/threshold_baseline.csv", index=False)
    df_rand.to_csv("results/random_baseline.csv", index=False)
    df_rl.to_csv("results/rl_agent.csv", index=False)
    print("Benchmarking complete. Results saved.")
if __name__ == "__main__":
    main()
