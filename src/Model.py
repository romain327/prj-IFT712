import os
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")

try:
    from utils.Metrics import Metrics
except ImportError:
    from utils.Metrics import Metrics


##
# @class Model
# @brief Classe abstraite de base pour tous les modèles de classification.
# @details Cette classe gère le cycle de vie complet des modèles : préparation des données,
#          entraînement, évaluation, tests, inférence unique et persistance (sauvegarde/chargement).
# @extends ABC
class Model(ABC):
    ##
    # @brief Constructeur de la classe Model.
    # @param n_train (int) Nombre d'échantillons à utiliser pour l'entraînement.
    # @param n_val (int) Nombre d'échantillons à utiliser pour la validation.
    # @param model_name (str) Nom identifiant du modèle (par défaut "BaseModel").
    # @note Définit le chemin de sauvegarde automatique dans le dossier `_models/`.
    def __init__(self, n_train, n_val, model_name="BaseModel"):
        self.n_train = n_train
        self.n_val = n_val
        self.model_name = model_name
        self.model_path = f"_models/{model_name}.pkl"
        self.model = None

    ##
    # @brief Méthode abstraite pour l'instanciation du modèle sous-jacent.
    # @param kwargs Arguments variables pour la configuration spécifique du modèle.
    # @return L'objet modèle brut (sklearn ou autre) non entraîné.
    # @note Doit être implémentée par toutes les classes filles.
    @abstractmethod
    def _create_model(self, **kwargs):
        pass

    ##
    # @brief Prépare et divise les données brutes en ensembles d'entraînement et de validation.
    # @param data_dict (dict) Dictionnaire contenant les clés "train_data" et "train_labels".
    # @return tuple (X_train, X_val, y_train, y_val) Les données divisées et stratifiées.
    def _prepare_data(self, data_dict):
        X = np.array(data_dict["train_data"])
        y = np.array(data_dict["train_labels"])

        limit = min(len(X), self.n_train + self.n_val)
        X_sub = X[:limit]
        y_sub = y[:limit]

        print(f"[{self.model_name}] Data split: {self.n_train} Train, {self.n_val} Val")

        return train_test_split(
            X_sub,
            y_sub,
            train_size=self.n_train,
            test_size=self.n_val,
            random_state=42,
            stratify=y_sub,
        )

    def _generate_cm_figure(self, cm, class_names):
        fig = plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.title(f"Confusion Matrix - {self.model_name}")
        plt.tight_layout()
        return fig

    ##
    # @brief Exécute le pipeline d'entraînement complet.
    # @details Prépare les données, redimensionne (flatten) si nécessaire, entraîne le modèle,
    #          et évalue les performances sur le set de validation.
    # @param data_dict (dict) Dictionnaire des données d'entraînement et labels.
    # @param class_names (list, optional) Liste des noms de classes pour l'annotation de la matrice de confusion.
    # @return tuple (model, metrics) Le modèle entraîné et un dictionnaire de métriques de validation.
    def train(self, data_dict: dict, class_names=None) -> tuple:
        print(f"[{self.model_name}] Preparing data...")
        X_train, X_val, y_train, y_val = self._prepare_data(data_dict)

        # Flatten automatique sauf pour le CNN
        if len(X_train.shape) > 2 and self.model_name != "CNN":
            X_train = X_train.reshape(X_train.shape[0], -1)
            X_val = X_val.reshape(X_val.shape[0], -1)

        self.model = self._create_model()
        print(f"[{self.model_name}] Training started...")

        self.model.fit(X_train, y_train)

        print(f"[{self.model_name}] Evaluating on validation set...")
        y_pred = self.model.predict(X_val)

        metrics = Metrics.calculate_metrics(y_val, y_pred, model_name=self.model_name)

        if "confusion_matrix" in metrics and class_names is not None:
            metrics["confusion_matrix_fig"] = self._generate_cm_figure(
                metrics["confusion_matrix"], class_names
            )
        else:
            metrics["confusion_matrix_fig"] = None

        print(
            f"[{self.model_name}] Validation Accuracy: {metrics.get('accuracy', 0):.4f}"
        )
        self.save_model()

        return self.model, metrics

    ##
    # @brief Évalue le modèle sur un jeu de données de test indépendant.
    # @details Charge le modèle sauvegardé, effectue les prédictions et calcule les métriques.
    # @param data_dict (dict) Dictionnaire contenant "test_data" et "test_labels".
    # @param class_names (list, optional) Liste des noms de classes pour la matrice de confusion.
    # @return dict Dictionnaire contenant les métriques de performance (accuracy, f1, confusion matrix, etc.).
    def test(self, data_dict: dict, class_names=None) -> dict:
        self.load_model()
        X_test = np.array(data_dict["test_data"])
        y_test = np.array(data_dict["test_labels"])

        if len(X_test.shape) > 2 and self.model_name != "CNN":
            X_test = X_test.reshape(X_test.shape[0], -1)

        y_pred = self.model.predict(X_test)
        metrics = Metrics.calculate_metrics(y_test, y_pred, model_name=self.model_name)

        if "confusion_matrix" in metrics and class_names is not None:
            metrics["confusion_matrix_fig"] = self._generate_cm_figure(
                metrics["confusion_matrix"], class_names
            )

        return metrics

    ##
    # @brief Effectue une inférence sur une image unique.
    # @param image (np.ndarray) L'image d'entrée (vecteur ou matrice).
    # @return int La classe prédite par le modèle.
    def classify(self, image: np.ndarray):
        if self.model is None:
            self.load_model()
        image = np.array(image)

        # Gestion du format d'entrée
        if len(image.shape) > 1 and self.model_name != "CNN":
            image = image.reshape(1, -1)
        elif len(image.shape) == 1 and self.model_name != "CNN":
            image = image.reshape(1, -1)

        return self.model.predict(image)[0]

    ##
    # @brief Sauvegarde l'instance du modèle sur le disque via pickle.
    # @note Le fichier est stocké dans `_models/{model_name}.pkl`.
    def save_model(self) -> None:
        os.makedirs("_models", exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump(self.model, f)
        print(f"[{self.model_name}] Model saved to {self.model_path}")

    ##
    # @brief Charge le modèle depuis le disque s'il existe.
    # @throws FileNotFoundError Si le fichier du modèle n'existe pas.
    def load_model(self) -> None:
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"{self.model_path} not found")
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)
