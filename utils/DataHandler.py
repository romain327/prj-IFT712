import os
import pickle
import numpy as np


##
# @class DataHandler
# @brief Gestionnaire de données pour le chargement et le prétraitement du dataset CIFAR-10.
# @details Cette classe s'occupe de lire les fichiers binaires (pickle), de charger les données d'entraînement et de test,
#          de normaliser les pixels et de redimensionner les images selon les besoins des modèles.
class DataHandler:

    ##
    # @brief Initialise le gestionnaire de données.
    # @param base_path (str) Chemin vers le dossier contenant les fichiers du dataset (data_batch_x).
    # @throws FileNotFoundError Si le dossier spécifié n'existe pas.
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.data = None
        self.train_data = None
        self.train_labels = None
        self.test_data = None
        self.test_labels = None

        self.class_names = [
            "Airplane",
            "Automobile",
            "Bird",
            "Cat",
            "Deer",
            "Dog",
            "Frog",
            "Horse",
            "Ship",
            "Truck",
        ]

        if not os.path.exists(base_path):
            raise FileNotFoundError(f"Directory {base_path} does not exist!")

    ##
    # @brief Méthode utilitaire interne pour désérialiser un fichier pickle.
    # @param filename (str) Nom du fichier à charger (relatif au base_path).
    # @return dict Le contenu brut du fichier pickle.
    # @throws FileNotFoundError Si le fichier n'est pas trouvé.
    def unpickle(self, filename: str) -> dict:
        filepath = os.path.join(self.base_path, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File {filepath} does not exist!")

        with open(filepath, "rb") as fo:
            raw_data = pickle.load(fo, encoding="bytes")

        return raw_data

    ##
    # @brief Charge l'intégralité du dataset (Train + Test) en mémoire.
    # @details Lit les 5 batches d'entraînement et le batch de test.
    # @param normalize (bool) Si True, divise les valeurs des pixels par 255.0 (float). Sinon, garde les valeurs brutes.
    # @param flatten (bool) Si True, retourne les images sous forme de vecteurs (N, 3072).
    #        Si False, retourne les images sous forme de tenseurs (N, 3, 32, 32).
    # @return dict Dictionnaire complet contenant "train_data", "train_labels", "test_data", "test_labels".
    def load_data(self, normalize: bool = True, flatten: bool = True) -> dict:
        train_files = [f"data_batch_{i}" for i in range(1, 6)]
        test_file = "test_batch"

        train_data = []
        train_labels = []

        print("Loading training data...")
        for filename in train_files:
            batch = self.unpickle(filename)
            train_data.append(batch[b"data"])
            train_labels.extend(batch[b"labels"])

        # Empiler dans un seul tableau
        train_data = np.vstack(train_data)  # shape (50000, 3072)
        train_labels = np.array(train_labels, dtype=np.int64)

        print("Loading test data...")
        test_batch = self.unpickle(test_file)

        test_data = np.array(test_batch[b"data"])
        test_labels = np.array(test_batch[b"labels"], dtype=np.int64)

        # Redimensionner les images à (N, 3, 32, 32) si nécessaire
        if not flatten:
            train_data = train_data.reshape(-1, 3, 32, 32)
            test_data = test_data.reshape(-1, 3, 32, 32)

        # Normaliser les valeurs de pixels si demandé
        if normalize:
            train_data = train_data.astype(np.float32) / 255.0
            test_data = test_data.astype(np.float32) / 255.0
        else:
            train_data = train_data.astype(np.float32)
            test_data = test_data.astype(np.float32)

        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels

        self.data = {
            "train_data": train_data,
            "train_labels": train_labels,
            "test_data": test_data,
            "test_labels": test_labels,
        }

        print(
            f"Data loaded: {len(train_data)} train samples, {len(test_data)} test samples"
        )

        return self.data

    ##
    # @brief Retourne les données d'entraînement (X, y).
    # @return tuple (train_data, train_labels).
    # @throws ValueError Si load_data() n'a pas été appelé au préalable.
    def get_train_data(self) -> tuple:
        if self.train_data is None:
            raise ValueError("Data not loaded yet. Call load_data() first.")
        return self.train_data, self.train_labels

    ##
    # @brief Retourne les données de test (X, y).
    # @return tuple (test_data, test_labels).
    # @throws ValueError Si load_data() n'a pas été appelé au préalable.
    def get_test_data(self) -> tuple:
        if self.test_data is None:
            raise ValueError("Data not loaded yet. Call load_data() first.")
        return self.test_data, self.test_labels

    ##
    # @brief Retourne le dictionnaire complet des données chargées.
    # @return dict Dictionnaire avec clés 'train_data', 'train_labels', 'test_data', 'test_labels'.
    # @throws ValueError Si load_data() n'a pas été appelé au préalable.
    def get_data_dict(self) -> dict:
        if self.data is None:
            raise ValueError("Data not loaded yet. Call load_data() first.")
        return self.data

    ##
    # @brief Extrait un sous-ensemble des données chargées (utile pour le débogage ou les tests rapides).
    # @param n_train (int, optional) Nombre d'échantillons d'entraînement à conserver.
    # @param n_test (int, optional) Nombre d'échantillons de test à conserver.
    # @return dict Dictionnaire contenant les sous-ensembles de données.
    # @throws ValueError Si les données n'ont pas encore été chargées.
    def get_subset(self, n_train: int = None, n_test: int = None) -> dict:
        if self.data is None:
            raise ValueError("Data not loaded yet. Call load_data() first.")

        train_data = self.train_data[:n_train] if n_train else self.train_data
        train_labels = self.train_labels[:n_train] if n_train else self.train_labels
        test_data = self.test_data[:n_test] if n_test else self.test_data
        test_labels = self.test_labels[:n_test] if n_test else self.test_labels

        return {
            "train_data": train_data,
            "train_labels": train_labels,
            "test_data": test_data,
            "test_labels": test_labels,
        }

    ##
    # @brief Calcule et affiche la distribution des classes dans un jeu de données.
    # @param dataset (str) "train" pour les données d'entraînement, "test" pour les données de test.
    # @return dict Dictionnaire associant chaque nom de classe à son nombre d'occurrences.
    # @throws ValueError Si les données ne sont pas chargées.
    def get_class_distribution(self, dataset: str = "train") -> dict:
        if self.data is None:
            raise ValueError("Data not loaded yet. Call load_data() first.")

        labels = self.train_labels if dataset == "train" else self.test_labels
        unique, counts = np.unique(labels, return_counts=True)

        class_names = [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]

        distribution = {class_names[i]: count for i, count in zip(unique, counts)}

        print(f"\nClass distribution ({dataset}):")
        for class_name, count in distribution.items():
            print(f"  {class_name:12s}: {count:5d} ({count / len(labels) * 100:.1f}%)")

        return distribution

    ##
    # @brief Récupère un échantillon unique par son index.
    # @param index (int) L'index de l'échantillon.
    # @param dataset (str) "train" ou "test".
    # @return tuple (image, label).
    def get_sample(self, index: int, dataset: str = "train") -> tuple:
        if self.data is None:
            raise ValueError("Data not loaded yet. Call load_data() first.")

        if dataset == "train":
            return self.train_data[index], self.train_labels[index]
        else:
            return self.test_data[index], self.test_labels[index]

    ##
    # @brief Récupère un lot d'échantillons donnés par une liste d'indices.
    # @param indices (list) Liste des indices à récupérer.
    # @param dataset (str) "train" ou "test".
    # @return tuple (images, labels).
    def get_batch(self, indices: list, dataset: str = "train") -> tuple:
        if self.data is None:
            raise ValueError("Data not loaded yet. Call load_data() first.")

        if dataset == "train":
            return self.train_data[indices], self.train_labels[indices]
        else:
            return self.test_data[indices], self.test_labels[indices]

    ##
    # @brief Retourne la liste des noms lisibles des classes (ex: "Airplane", "Bird"...).
    # @return list Liste de chaînes de caractères.
    def get_class_names(self):
        return self.class_names
