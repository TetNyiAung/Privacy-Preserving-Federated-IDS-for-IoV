import pandas as pd
import matplotlib.pyplot as plt
import os

CSV_FILE = "results/results_log.csv"
PLOTS_DIR = "plots"

# Check if results file exists
if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(f"{CSV_FILE} not found. Please run the server first.")

# Create output folder
os.makedirs(PLOTS_DIR, exist_ok=True)

# Load results
df = pd.read_csv(CSV_FILE)
metrics = ["Accuracy", "Precision", "Recall", "F1"]

# Generate and save individual plots
for metric in metrics:
    plt.figure(figsize=(8, 5))
    plt.plot(df["round"], df[metric], marker='o', linestyle='-', linewidth=2, markersize=6)
    plt.title(f"{metric} Over Federated Rounds", fontsize=14, fontweight='bold')
    plt.xlabel("Federated Round", fontsize=12)
    plt.ylabel(metric, fontsize=12)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(df["round"])
    plt.tight_layout()
    
    filename = os.path.join(PLOTS_DIR, f"{metric.lower()}_plot.png")
    plt.savefig(filename, dpi=300)
    plt.close()

print(f"All metric plots saved in '{PLOTS_DIR}/'")
