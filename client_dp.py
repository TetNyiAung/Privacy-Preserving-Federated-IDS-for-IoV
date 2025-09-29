import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from model import IntrusionDetectionModel
from utils import get_partitioned_data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLIENTS, TEST_SET = get_partitioned_data(num_clients=10)

def apply_dp(parameters, epsilon=1.0):
    noise_scale = 1.0 / epsilon
    return [param + np.random.normal(0.0, noise_scale, param.shape) for param in parameters]

class VehicleClientDP(fl.client.NumPyClient):
    def __init__(self, cid: int):
        self.cid = f"client_{cid}"
        self.X, self.y = CLIENTS[self.cid]
        self.X = torch.tensor(self.X, dtype=torch.float32).to(DEVICE)
        self.y = torch.tensor(self.y.values, dtype=torch.float32).to(DEVICE).view(-1, 1)

        input_dim = self.X.shape[1]
        self.model = IntrusionDetectionModel(input_dim).to(DEVICE)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def get_parameters(self, config=None):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        new_state_dict = {k: torch.tensor(v) for k, v in zip(state_dict.keys(), parameters)}
        self.model.load_state_dict(new_state_dict, strict=True)

    def fit(self, parameters, config=None):
        try:
            self.set_parameters(parameters)
            self.model.train()

            if self.cid == "client_1":
                print(f"[{self.cid}] Label Flipping attack.")
                y_train = 1 - self.y
            else:
                y_train = self.y

            dataset = TensorDataset(self.X, y_train)
            loader = DataLoader(dataset, batch_size=32, shuffle=True)

            for _ in range(5):
                for X_batch, y_batch in loader:
                    self.optimizer.zero_grad()
                    output = self.model(X_batch)
                    loss = self.criterion(output, y_batch)
                    loss.backward()
                    self.optimizer.step()

            if self.cid == "client_2":
                print(f"[{self.cid}] Model Poisoning attack.")
                with torch.no_grad():
                    for param in self.model.parameters():
                        param.add_(torch.randn_like(param) * 1.0)

            updated_params = self.get_parameters()
            if self.cid == "client_9":
                print(f"[{self.cid}] Noise Injection attack.")
                updated_params = [param + np.random.normal(0, 1.0, param.shape) for param in updated_params]

            dp_params = apply_dp(updated_params, epsilon=0.5)

            self.model.eval()
            with torch.no_grad():
                output = self.model(self.X)
                preds = (torch.sigmoid(output) > 0.5).float()
                acc = (preds == self.y).float().mean().item()

            print(f"[{self.cid}] Training done with DP (ε=1.0)")
            return dp_params, len(self.X), {
                "loss": float(loss.item()),
                "accuracy": acc,
                "cid": self.cid
            }

        except Exception as e:
            print(f"[{self.cid}] Error in fit(): {e}")
            fallback_params = self.get_parameters()
            return fallback_params, len(self.X), {
                "loss": 9999.0,
                "accuracy": None,
                "cid": self.cid
            }

    def evaluate(self, parameters, config=None):
        self.set_parameters(parameters)
        self.model.eval()
        with torch.no_grad():
            out = self.model(self.X)
            loss = self.criterion(out, self.y)
            pred = (torch.sigmoid(out) > 0.5).float()
            acc = (pred == self.y).float().mean().item()
        return float(loss.item()), len(self.X), {
            "accuracy": acc,
            "cid": self.cid
        }

def run_client(cid):
    client = VehicleClientDP(cid)
    fl.client.start_numpy_client(server_address="localhost:8080", client=client)
