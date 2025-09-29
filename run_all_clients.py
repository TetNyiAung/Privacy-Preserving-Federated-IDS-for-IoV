import subprocess
import time

NUM_CLIENTS = 10
client_processes = []

for cid in range(1, NUM_CLIENTS + 1):
    print(f"Launching Client {cid}...")
    process = subprocess.Popen(
        ["python", "-c", f"from client_dp import run_client; run_client({cid})"]
        # REMOVE these two lines for debugging:
        # , stdout=subprocess.DEVNULL
        # , stderr=subprocess.DEVNULL
    )
    client_processes.append(process)
    time.sleep(2.0)
