from utils import get_partitioned_data

clients, test_set = get_partitioned_data()

for cid, (X, y) in clients.items():
    print(f"{cid} has {len(X)} samples")