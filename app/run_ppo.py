import os
import pandas as pd
import subprocess

def run_ppo():
    # Run PPO training and return the path to the results file
    subprocess.run(["python", "experiments/train_ppo.py", "--timesteps", "20000", "--eval-episodes", "5"])
    return None
