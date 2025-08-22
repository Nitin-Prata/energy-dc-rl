import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ===============================
# 1. Load & Preprocess Data
# ===============================
def load_data(file_path):
    """Loads CSV data into a DataFrame."""
    df = pd.read_csv(file_path)
    df.dropna(inplace=True)
    return df

# ===============================
# 2. Compute Rolling Average
# ===============================
def compute_rolling_average(data, column, window=5):
    """Computes rolling average for smoothing."""
    return data[column].rolling(window=window).mean()

# ===============================
# 3. Plotting Function
# ===============================
def plot_data(df, x_col, y_col, rolling_col=None):
    """Plots raw data and optional rolling average."""
    plt.figure(figsize=(10, 6))
    
    # Plot raw data
    plt.plot(df[x_col], df[y_col], label="Raw Data", color="blue", alpha=0.6)
    
    # Plot rolling average if available
    if rolling_col:
        plt.plot(df[x_col], df[rolling_col], label=f"{rolling_col} (Smoothed)", color="red", linewidth=2)
    
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(y_col, fontsize=12)
    plt.title(f"{y_col} Over {x_col}", fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

# ===============================
# 4. Main Execution
# ===============================
if __name__ == "__main__":
    file_path = "data.csv"  # Change to your file path
    df = load_data(file_path)
    
    # Add rolling average
    df["Smoothed"] = compute_rolling_average(df, "Value", window=5)
    
    # Plot
    plot_data(df, "Time", "Value", rolling_col="Smoothed")
