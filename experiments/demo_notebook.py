# RL Data Center Demo Notebook
import pandas as pd
import matplotlib.pyplot as plt
# Load results
metrics = pd.read_csv('runs/metrics.csv')
plt.figure(figsize=(10,5))
plt.plot(metrics['episode'], metrics['reward'], label='Reward')
plt.plot(metrics['episode'], metrics['energy'], label='Energy')
plt.plot(metrics['episode'], metrics['total_cost'], label='Cost')
plt.plot(metrics['episode'], metrics['total_carbon'], label='Carbon')
plt.xlabel('Episode')
plt.ylabel('Metric Value')
plt.legend()
plt.title('RL Agent Performance Over Episodes')
plt.show()
# Compare baselines
thresh = pd.read_csv('results/threshold_baseline.csv')
rand = pd.read_csv('results/random_baseline.csv')
rl = pd.read_csv('results/rl_agent.csv')
plt.figure(figsize=(10,5))
plt.bar(['Threshold', 'Random', 'RL'], [thresh['reward'].mean(), rand['reward'].mean(), rl['reward'].mean()])
plt.ylabel('Average Reward')
plt.title('Baseline vs RL Agent')
plt.show()
