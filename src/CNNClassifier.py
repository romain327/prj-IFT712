import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from src.Model import Model
from utils.Metrics import Metrics
from utils.TqdmToLogger import TqdmToLogger


##
# @class CNNArchitecture
# @brief Architecture du réseau de neurones convolutif (CNN) définie avec PyTorch.
# @extends torch.nn.Module
# @details Comprend 3 blocs de convolution (Conv2d -> BatchNorm -> ReLU -> MaxPool -> Dropout)
#          suivis de couches dense (Fully Connected). Conçu pour des entrées image 32x32x3.
class CNNArchitecture(nn.Module):

    def __init__(self):
        super(CNNArchitecture, self).__init__()
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

    ##
    # @brief Définit la passe avant (forward pass) du réseau.
    # @param x (torch.Tensor) Tenseur d'entrée (Batch, 3, 32, 32).
    # @return torch.Tensor Logits de sortie (Batch, 10).
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


##
# @class CNNClassifier
# @brief Adaptateur pour le modèle CNN PyTorch intégrant la structure de la classe Model.
# @extends Model
# @details Gère spécifiquement les tenseurs PyTorch, l'utilisation du GPU (CUDA),
#          la boucle d'entraînement manuelle et l'Early Stopping.
class CNNClassifier(Model):
    def __init__(self, logger, n_train, n_val):
        super().__init__(logger, n_train, n_val, model_name="CNN")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = "_models/CNN.pth"
        # Structure pour stocker l'historique (compatible avec votre snippet)
        self.history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
        self.tqdm_out = TqdmToLogger(self.logger, "TQDM")

    def _create_model(self, **kwargs):
        return CNNArchitecture().to(self.device)

    ##
    # @brief Surcharge de la méthode d'entraînement pour PyTorch.
    # @details Gère :
    #          - La conversion des données Numpy vers des Tenseurs PyTorch (permute channels).
    #          - La création des DataLoaders.
    #          - La boucle d'entraînement sur 50 époques avec barre de progression (tqdm).
    #          - L'optimiseur Adam et le Scheduler ReduceLROnPlateau.
    #          - L'Early Stopping basé sur la `val_loss`.
    #          - La sauvegarde de l'historique d'apprentissage.
    # @param data_dict (dict) Données d'entraînement.
    # @param class_names (list, optional) Noms des classes.
    # @return tuple (model, metrics) Le modèle PyTorch (nn.Module) et les métriques finales.
    def train(self, data_dict: dict, class_names=None) -> tuple:
        self.logger.log(f"[{self.model_name}] Using device: {self.device}", "INFO")

        X_train_raw, X_val_raw, y_train, y_val = self._prepare_data(data_dict)

        if len(X_train_raw.shape) == 2:
            X_train_raw = X_train_raw.reshape(-1, 32, 32, 3)
            X_val_raw = X_val_raw.reshape(-1, 32, 32, 3)

        X_train = torch.FloatTensor(X_train_raw).permute(0, 3, 1, 2)
        y_train_t = torch.LongTensor(y_train)

        X_val = torch.FloatTensor(X_val_raw).permute(0, 3, 1, 2)
        y_val_t = torch.LongTensor(y_val)

        train_loader = DataLoader(
            TensorDataset(X_train, y_train_t), batch_size=32, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(X_val, y_val_t), batch_size=32, shuffle=False
        )

        # Initialisation Modèle, Optimiseur, Scheduler, EarlyStopping
        self.model = self._create_model()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=3
        )

        # Paramètres Early Stopping
        patience = 5
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        self.logger.log(f"[{self.model_name}] Training started (Epochs=50)...", "INFO")

        for epoch in range(50):
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            train_pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1:2d}/50 [Train]",
                position=0,
                leave=True,
                ncols=100,
                file=self.tqdm_out,
                mininterval=1.0,
            )

            for batch_X, batch_y in train_pbar:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_X)
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

            # Phase de Validation
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            val_pbar = tqdm(
                val_loader,
                desc=f"Epoch {epoch + 1:2d}/50 [Val]  ",
                position=0,
                leave=True,
                ncols=100,
                file=self.tqdm_out,
                mininterval=1.0,
            )

            with torch.no_grad():
                for batch_X, batch_y in val_pbar:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_X)
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

            # Mise à jour historique
            self.history["loss"].append(train_loss)
            self.history["accuracy"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_accuracy"].append(val_acc)

            # Scheduler step
            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(val_loss)

            self.logger.log(
                f"Epoch {epoch + 1:2d}/50 Summary - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | LR: {current_lr:.2e}",
                "RESULT",
            )

            # Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered after epoch {epoch + 1}")
                    break

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            self.logger.log("Restored best model based on validation loss.", "INFO")

        # 4. Évaluation Finale pour le retour (Compatible Controller)
        self.logger.log(f"[{self.model_name}] Generating final metrics...", "INFO")
        val_preds = self._predict_batch(X_val, y_val_t)

        metrics = Metrics.calculate_metrics(
            self.logger, y_val, val_preds, model_name=self.model_name
        )

        metrics["history_fig"] = Metrics.plot_training_history(self.history, "cnn")

        metrics["loss"] = best_val_loss

        if "confusion_matrix" in metrics and class_names is not None:
            metrics["confusion_matrix_fig"] = self._generate_cm_figure(
                metrics["confusion_matrix"], class_names
            )
        else:
            metrics["confusion_matrix_fig"] = None

        self.save_model()

        return self.model, metrics

    ##
    # @brief Surcharge de la méthode de test pour PyTorch.
    # @details Convertit les données de test en tenseurs et effectue l'inférence par batch
    #          pour éviter les erreurs de mémoire (OOM).
    # @param data_dict (dict) Données de test.
    # @param class_names (list, optional) Noms des classes.
    # @return dict Métriques de performance sur le jeu de test.
    def test(self, data_dict: dict, class_names=None) -> dict:
        self.load_model()

        X_test = np.array(data_dict["test_data"])
        y_test = np.array(data_dict["test_labels"])

        # Reshape si nécessaire
        if len(X_test.shape) == 2:
            X_test = X_test.reshape(-1, 32, 32, 3)

        # Conversion Tenseurs
        X_test_t = torch.FloatTensor(X_test).permute(0, 3, 1, 2)
        y_test_t = torch.LongTensor(y_test)

        self.logger.log(
            f"[{self.model_name}] Testing on {len(X_test)} samples...", "INFO"
        )

        # Inférence par batch
        test_preds = self._predict_batch(X_test_t, y_test_t)

        # Calcul métriques
        metrics = Metrics.calculate_metrics(
            self.logger, y_test, test_preds, model_name=self.model_name
        )

        if "confusion_matrix" in metrics and class_names is not None:
            metrics["confusion_matrix_fig"] = self._generate_cm_figure(
                metrics["confusion_matrix"], class_names
            )

        return metrics

    ##
    # @brief Méthode utilitaire interne pour prédire sur un grand jeu de données par lots.
    # @param X_tensor (torch.Tensor) Données d'entrée.
    # @param y_tensor (torch.Tensor) Labels (utilisés uniquement pour créer le Dataset compatible).
    # @return np.ndarray Tableau numpy des classes prédites.
    def _predict_batch(self, X_tensor, y_tensor):
        self.model.eval()
        loader = DataLoader(
            TensorDataset(X_tensor, y_tensor), batch_size=32, shuffle=False
        )
        all_preds = []

        with torch.no_grad():
            # Une simple barre de progression pour l'inférence
            for batch_X, _ in tqdm(
                loader,
                desc="Inference",
                unit="batch",
                leave=False,
                file=self.tqdm_out,
                mininterval=1.0,
            ):
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                preds = outputs.argmax(1).cpu().numpy()
                all_preds.extend(preds)

        return np.array(all_preds)

    ##
    # @brief Inférence sur une seule image avec PyTorch.
    # @details Gère le redimensionnement (32, 32, 3) et le passage sur le device (CPU/GPU).
    # @param image (np.ndarray) Image brute.
    # @return int Indice de la classe prédite.
    def classify(self, image: np.ndarray) -> int:
        self.load_model()
        self.model.eval()

        image = np.array(image)
        if len(image.shape) == 1:
            image = image.reshape(32, 32, 3)

        image_t = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image_t)
            predicted_class = outputs.argmax(1).item()

        return predicted_class

    ##
    # @brief Sauvegarde les poids du modèle (state_dict) via torch.save.
    def save_model(self) -> None:
        os.makedirs("_models", exist_ok=True)
        torch.save(self.model.state_dict(), self.model_path)
        self.logger.log(f"[{self.model_name}] Model saved to {self.model_path}", "INFO")

    ##
    # @brief Charge les poids du modèle (state_dict) via torch.load.
    # @details Assure le mapping correct sur le device (CPU/GPU).
    def load_model(self) -> None:
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"{self.model_path} not found")

        if self.model is None:
            self.model = self._create_model()

        self.model.load_state_dict(
            torch.load(self.model_path, map_location=self.device)
        )
