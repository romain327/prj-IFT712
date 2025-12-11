import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

model_path = "_models/CNN.pth"


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.3)

        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.4)

        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(0.5)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.bn_fc = nn.BatchNorm1d(128)
        self.dropout_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1_1(self.conv1_1(x)))
        x = self.relu(self.bn1_2(self.conv1_2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.relu(self.bn2_1(self.conv2_1(x)))
        x = self.relu(self.bn2_2(self.conv2_2(x)))
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.relu(self.bn3_1(self.conv3_1(x)))
        x = self.relu(self.bn3_2(self.conv3_2(x)))
        x = self.pool3(x)
        x = self.dropout3(x)

        x = self.flatten(x)
        x = self.relu(self.bn_fc(self.fc1(x)))
        x = self.dropout_fc(x)
        x = self.fc2(x)

        return x


def create_model(device):
    model = CNNModel().to(device)
    return model


def train(data_dict, n_train, n_val):
    print("=== CNN Classifier (PyTorch) ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X = data_dict["train_data"]
    y = np.array(data_dict["train_labels"])

    if len(X.shape) == 2:
        X = X.reshape(-1, 32, 32, 3)

    X_sub = X[: n_train + n_val]
    y_sub = y[: n_train + n_val]

    X_train = X_sub[:n_train]
    y_train = y_sub[:n_train]
    X_val = X_sub[n_train:]
    y_val = y_sub[n_train:]

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    print(f"Image shape: {X_train.shape[1:]}")

    X_train = torch.FloatTensor(X_train).permute(0, 3, 1, 2)
    y_train = torch.LongTensor(y_train)
    X_val = torch.FloatTensor(X_val).permute(0, 3, 1, 2)
    y_val = torch.LongTensor(y_val)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    model = create_model(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-7
    )

    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

    print("\nTraining model...")
    start_train = time.time()

    best_val_loss = float("inf")
    patience_counter = 0
    patience = 10
    best_model_state = None

    for epoch in range(50):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1:2d}/50 [Train]",
            position=0,
            leave=True,
            ncols=100,
        )

        for batch_X, batch_y in train_pbar:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_X.size(0)
            _, predicted = outputs.max(1)
            train_total += batch_y.size(0)
            train_correct += predicted.eq(batch_y).sum().item()

            train_pbar.set_postfix(
                {
                    "loss": f"{loss.item():.3f}",
                    "acc": f"{train_correct / train_total:.3f}",
                }
            )

        train_pbar.close()

        train_loss = train_loss / train_total
        train_acc = train_correct / train_total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        val_pbar = tqdm(
            val_loader,
            desc=f"Epoch {epoch + 1:2d}/50 [Val]  ",
            position=0,
            leave=True,
            ncols=100,
        )

        with torch.no_grad():
            for batch_X, batch_y in val_pbar:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item() * batch_X.size(0)
                _, predicted = outputs.max(1)
                val_total += batch_y.size(0)
                val_correct += predicted.eq(batch_y).sum().item()

                val_pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.3f}",
                        "acc": f"{val_correct / val_total:.3f}",
                    }
                )

        val_pbar.close()

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        history["loss"].append(train_loss)
        history["accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch + 1:2d}/50 Summary - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after epoch {epoch + 1}")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    train_time = time.time() - start_train
    print(
        f"\nTraining completed in {train_time:.1f} seconds "
        f"({train_time / 60:.1f} minutes)"
    )

    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            val_loss += loss.item() * batch_X.size(0)
            _, predicted = outputs.max(1)
            val_total += batch_y.size(0)
            val_correct += predicted.eq(batch_y).sum().item()

    val_loss = val_loss / val_total
    val_score = val_correct / val_total

    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    plot_training_history(history)

    return val_score, val_loss


def test(data_dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file {model_path} not found. "
            "Make sure you have trained the model before testing."
        )

    model = create_model(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    X_test = data_dict["test_data"]
    y_test = np.array(data_dict["test_labels"])

    if len(X_test.shape) == 2:
        X_test = X_test.reshape(-1, 32, 32, 3)

    X_test = torch.FloatTensor(X_test).permute(0, 3, 1, 2)
    y_test = torch.LongTensor(y_test)

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            test_loss += loss.item() * batch_X.size(0)
            _, predicted = outputs.max(1)
            test_total += batch_y.size(0)
            test_correct += predicted.eq(batch_y).sum().item()

    test_loss = test_loss / test_total
    test_score = test_correct / test_total

    return test_score, test_loss


def classify(image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file {model_path} not found. "
            "Make sure you have trained the model before calling classify()."
        )

    model = create_model(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    if len(image.shape) == 1:
        image = image.reshape(32, 32, 3)

    image = image.astype("float32") / 255.0
    image = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        predicted_class = outputs.argmax(1).item()

    return predicted_class


def get_probabilities(image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file {model_path} not found. "
            "Make sure you have trained the model before calling get_probabilities()."
        )

    model = create_model(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    if len(image.shape) == 1:
        image = image.reshape(32, 32, 3)

    image = image.astype("float32") / 255.0
    image = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()

    return probabilities


def plot_training_history(history, save_path="_results/loss_accuracy_pt.png"):
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["accuracy"]) + 1)

    ax1.plot(epochs, history["accuracy"], "b-", label="Train Accuracy", linewidth=2)
    ax1.plot(epochs, history["val_accuracy"], "r-", label="Val Accuracy", linewidth=2)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.set_title("CNN - Accuracy", fontsize=14, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["loss"], "b-", label="Train Loss", linewidth=2)
    ax2.plot(epochs, history["val_loss"], "r-", label="Val Loss", linewidth=2)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Loss", fontsize=12)
    ax2.set_title("CNN - Loss", fontsize=14, fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Training history saved to: {save_path}")

    plt.show()
