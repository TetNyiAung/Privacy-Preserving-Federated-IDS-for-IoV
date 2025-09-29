import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_toniot_dataset(path="data/train_test_network.csv"):
    df = pd.read_csv(path)
    df.dropna(inplace=True)

    if 'label' not in df.columns:
        raise ValueError("Missing 'label' column")

    df['label'] = df['label'].apply(lambda x: 0 if str(x).lower() == 'normal' else 1)

    non_numeric = df.select_dtypes(include=['object']).columns
    df.drop(columns=[col for col in non_numeric if col != 'label'], inplace=True)

    X = df.drop(columns=['label'])
    y = df['label']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

def get_partitioned_data(num_clients=10):
    X, y = load_toniot_dataset("data/train_test_network.csv")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )

    X_split = np.array_split(X_train, num_clients)
    y_split = np.array_split(y_train, num_clients)

    clients = {}
    for i in range(num_clients):
        clients[f"client_{i+1}"] = (
            X_split[i],
            pd.Series(y_split[i]).reset_index(drop=True)
        )

    test_data = (pd.DataFrame(X_test), pd.Series(y_test).reset_index(drop=True))
    return clients, test_data
