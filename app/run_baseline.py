import os
import pandas as pd
from scripts.run_baseline import run_baseline as cli_run_baseline

def run_baseline():
    # Run the baseline and return the path to the results file
    cli_run_baseline(num_episodes=20, output_dir="results")
    return None
