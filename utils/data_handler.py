import os
import pickle
import numpy as np


def unpickle(base_path, filename):
    filepath = os.path.join(base_path, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File {filepath} does not exist!")

    with open(filepath, "rb") as fo:
        raw_data = pickle.load(fo, encoding="bytes")

    return raw_data


def load_all_data(base_path, normalize=True, flatten=True):
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Directory {base_path} does not exist!")

    train_files = [f"data_batch_{i}" for i in range(1, 6)]
    test_file = "test_batch"

    train_data = []
    train_labels = []

    print("Loading training data...")
    for filename in train_files:
        batch = unpickle(base_path, filename)

        train_data.append(batch[b"data"])
        train_labels.extend(batch[b"labels"])

    # Stack into a single array
    train_data = np.vstack(train_data)  # shape (50000, 3072)
    train_labels = np.array(train_labels, dtype=np.int64)

    print("Loading test data...")
    test_batch = unpickle(base_path, test_file)

    test_data = np.array(test_batch[b"data"])
    test_labels = np.array(test_batch[b"labels"], dtype=np.int64)

    # Reshape images to (N, 3, 32, 32) if needed
    if not flatten:
        train_data = train_data.reshape(-1, 3, 32, 32)
        test_data = test_data.reshape(-1, 3, 32, 32)

    # Normalize pixel values if requested
    if normalize:
        train_data = train_data.astype(np.float32) / 255.0
        test_data = test_data.astype(np.float32) / 255.0
    else:
        train_data = train_data.astype(np.float32)
        test_data = test_data.astype(np.float32)

    data = {
        "train_data": train_data,
        "train_labels": train_labels,
        "test_data": test_data,
        "test_labels": test_labels,
    }

    print(
        f"Data loaded: {len(train_data)} train samples, {len(test_data)} test samples"
    )

    return data
