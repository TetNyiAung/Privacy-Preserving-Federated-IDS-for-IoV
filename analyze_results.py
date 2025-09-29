import pandas as pd
import os

CSV_FILE = "results/results_log.csv"
TEXT_SUMMARY = "results/detailed_summary.txt"

if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(f"{CSV_FILE} not found. Please run the server first.")

# Load results
df = pd.read_csv(CSV_FILE)

# Round-wise table
round_table = df.round(3)

# Average metrics
avg_metrics = round_table[["Accuracy", "Precision", "Recall", "F1"]].mean().round(3)

# Write to text file with clean formatting
with open(TEXT_SUMMARY, "w", encoding="utf-8") as f:
    f.write("╔════════════════════════════════════════════╗\n")
    f.write("║        Federated IDS Performance Log       ║\n")
    f.write("╚════════════════════════════════════════════╝\n\n")

    f.write("Round-wise IDS Performance:\n")
    f.write("-" * 75 + "\n")
    f.write(f"{'Round':<8}{'TP':<6}{'FP':<6}{'FN':<6}{'TN':<6}{'Accuracy':<12}{'Precision':<12}{'Recall':<12}{'F1':<12}\n")
    f.write("-" * 75 + "\n")
    for _, row in round_table.iterrows():
        f.write(f"{int(row['round']):<8}{int(row['TP']):<6}{int(row['FP']):<6}{int(row['FN']):<6}{int(row['TN']):<6}"
                f"{row['Accuracy']:<12.3f}{row['Precision']:<12.3f}{row['Recall']:<12.3f}{row['F1']:<12.3f}\n")
    f.write("-" * 75 + "\n\n")

    f.write("Average Metrics Across All Rounds:\n")
    f.write("-" * 75 + "\n")
    f.write(f"{'Accuracy':<12}{'Precision':<12}{'Recall':<12}{'F1':<12}\n")
    f.write(f"{avg_metrics['Accuracy']:<12.3f}{avg_metrics['Precision']:<12.3f}"
            f"{avg_metrics['Recall']:<12.3f}{avg_metrics['F1']:<12.3f}\n")
    f.write("-" * 75 + "\n")

print(f"Text summary saved to '{TEXT_SUMMARY}'")
