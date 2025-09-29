import flwr as fl
import numpy as np
import pandas as pd
from termcolor import colored
import os
from scipy.spatial.distance import cosine  # For gradient anomaly detection
from flwr.common import parameters_to_ndarrays
from scipy.stats import trim_mean

NUM_ROUNDS = 7
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
CSV_FILE = os.path.join(RESULTS_DIR, "results_log.csv")
SUMMARY_FILE = os.path.join(RESULTS_DIR, "summary.txt")

MALICIOUS_CLIENTS = {
    "client_1": "Label Flipping",
    "client_2": "Model Poisoning",
    "client_9": "Noise Injection",
}

class CustomFedStrategy(fl.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.previous_weights = None  # For anomaly detection

    def aggregate_fit(self, server_round, results, failures):
        if server_round < 3:
            return super().aggregate_fit(server_round, results, failures)

        if not results:
            print(colored(f"[Round {server_round}] No successful client results", "red"))
            return None, {}

        client_ids, losses, accuracies = [], {}, {}
        client_weights = {}

        for client_proxy, fit_res in results:
            cid = fit_res.metrics.get("cid", str(client_proxy.cid))
            client_ids.append(cid)
            losses[cid] = fit_res.metrics.get("loss")
            accuracies[cid] = fit_res.metrics.get("accuracy")
            client_weights[cid] = fit_res.parameters

        print(colored(f"\n[Round {server_round}] Client Losses:", "blue"))
        for cid, loss in losses.items():
            print(f"  - {cid}: {loss:.4f}" if loss is not None else f"  - {cid}: None")

        print(colored(f"\n[Round {server_round}] Client Accuracies:", "magenta"))
        for cid, acc in accuracies.items():
            print(f"  - {cid}: {acc:.4f}" if acc is not None else f"  - {cid}: None")

        # IDS Detection
        flagged = set()

        # --- Rule 1: High Loss Anomaly Detection ---
        loss_values = [v for v in losses.values() if v is not None]
        if loss_values:
            mean_loss = np.mean(loss_values)
            std_loss = np.std(loss_values)
            loss_threshold = mean_loss + 0.5 * std_loss
            flagged.update([cid for cid, loss in losses.items() if loss is not None and loss > loss_threshold])

        # --- Rule 2: Low Accuracy Detection ---
        acc_values = [v for v in accuracies.values() if v is not None]
        if acc_values:
            mean_acc = np.mean(acc_values)
            std_acc = np.std(acc_values)
            acc_threshold = mean_acc - 0.5 * std_acc
            flagged.update([cid for cid, acc in accuracies.items() if acc is not None and acc < acc_threshold])
            flagged.update([cid for cid, acc in accuracies.items() if acc == 0.0])

        # --- Rule 3: Gradient/Weight Anomaly Detection ---
        if self.previous_weights is not None:
            prev = np.concatenate([p.flatten() for p in self.previous_weights])

            weight_updates = {}
            for cid, weights in client_weights.items():
                update = np.concatenate([p.flatten() for p in parameters_to_ndarrays(weights)])
                delta = update - prev
                weight_updates[cid] = delta

            updates = list(weight_updates.values())

            if updates:
                # Robust median update instead of trim_mean (faster, more stable)
                median_update = np.median(updates, axis=0)
                median_norm = np.linalg.norm(median_update)

                cosine_threshold = 0.60
                norm_multiplier = 4.0

                for cid, update in weight_updates.items():
                    if cid in flagged:
                        continue  # Skip if already flagged for accuracy/loss

                    try:
                        sim = 1 - cosine(update, median_update)
                        norm = np.linalg.norm(update)

                        # Cosine similarity (direction)
                        if sim < cosine_threshold:
                            flagged.add(cid)

                        # Norm (magnitude)
                        if norm > median_norm * norm_multiplier:
                            flagged.add(cid)

                    except Exception as e:
                        print(f"[Warning] Gradient anomaly skipped for {cid}: {e}")

        # Save this round’s weights for next round comparison
        any_params = next(iter(client_weights.values()))
        self.previous_weights = parameters_to_ndarrays(any_params)

        # IDS Performance Summary
        final_flagged = list(flagged)
        ground_truth = [cid for cid in client_ids if cid in MALICIOUS_CLIENTS]
        trusted = [cid for cid in client_ids if cid not in final_flagged]

        TP = len([cid for cid in final_flagged if cid in ground_truth])
        FP = len([cid for cid in final_flagged if cid not in ground_truth])
        FN = len([cid for cid in ground_truth if cid not in final_flagged])
        TN = len([cid for cid in trusted if cid not in ground_truth])

        acc = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) else 0
        prec = TP / (TP + FP) if (TP + FP) else 0
        rec = TP / (TP + FN) if (TP + FN) else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0

        # Print summaries
        print(colored(f"\n[Round {server_round}] Ground Truth Malicious Clients:", "red"))
        for cid in ground_truth:
            print(f"   - {cid} ({MALICIOUS_CLIENTS[cid]})")

        print(colored(f"\n[Round {server_round}] Trusted Clients:", "green"))
        for cid in trusted:
            print(f"   - {cid} (Normal / Non-Attacker)")

        print(colored(f"\n[Round {server_round}] Flagged as Malicious:", "yellow"))
        for cid in final_flagged:
            attack = MALICIOUS_CLIENTS.get(cid, "Normal, False Positive")
            print(f"   - {cid} ({attack})")

        print(colored(f"\n[Round {server_round}] IDS Performance:", "cyan"))
        print(f"   TP={TP} | FP={FP} | FN={FN} | TN={TN}")
        print(f"   Accuracy={acc:.2f} | Precision={prec:.2f} | Recall={rec:.2f} | F1={f1:.2f}")

        # Log to CSV
        row = {
            "round": server_round,
            "TP": TP, "FP": FP, "FN": FN, "TN": TN,
            "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1,
        }

        df = pd.DataFrame([row])
        if not os.path.exists(CSV_FILE):
            df.to_csv(CSV_FILE, index=False)
        else:
            df.to_csv(CSV_FILE, mode="a", header=False, index=False)

        return super().aggregate_fit(server_round, results, failures)

def run_server():
    strategy = CustomFedStrategy(
        fraction_fit=1.0,
        min_fit_clients=10,
        min_available_clients=10,
    )

    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
            f.write("╔════════════════════════════════════════════════════════╗\n")
            f.write("║              Federated IDS Performance Summary         ║\n")
            f.write("╚════════════════════════════════════════════════════════╝\n\n")

            f.write("┌────────────────────────────────────────────────────────┐\n")
            f.write("│                  Round-wise Metrics                    │\n")
            f.write("└────────────────────────────────────────────────────────┘\n")
            f.write(df.round(2).to_string(index=False, justify="center"))
            f.write("\n\n")

            f.write("┌────────────────────────────────────────────────────────┐\n")
            f.write("│              Average Metrics Across Rounds             │\n")
            f.write("└────────────────────────────────────────────────────────┘\n")
            avg_df = df[["Accuracy", "Precision", "Recall", "F1"]].mean().round(2).to_frame("Mean").T
            f.write(avg_df.to_string(index=False, justify="center"))

if __name__ == "__main__":
    run_server()
