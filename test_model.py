from model import IntrusionDetectionModel
from utils import get_partitioned_data
import torch

clients, _ = get_partitioned_data()
X_sample, _ = clients["client_1"]
input_dim = X_sample.shape[1]

model = IntrusionDetectionModel(input_dim)
sample_input = torch.tensor(X_sample[:5], dtype=torch.float32)

output = model(sample_input)
print("Sample output:\n", output)