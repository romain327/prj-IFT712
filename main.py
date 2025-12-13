##
# @file main.py
# @brief Script principal d'orchestration pour la classification d'images CIFAR-10.
# @details Ce fichier sert de point d'entrée unique pour l'exécution du pipeline complet d'apprentissage automatique.
#          Il ne contient pas de classes mais exécute séquentiellement les étapes suivantes :
#          1. **Configuration** : Création des dossiers de sortie (`_models`, `_results`) et définition des paramètres.
#          2. **Chargement des données** : Utilisation de `DataHandler` pour importer et normaliser le dataset CIFAR-10.
#          3. **Instanciation** : Initialisation des différents modèles (DecisionTree, RandomForest, LogisticRegression, SVM, GBM, CNN).
#          4. **Entraînement** : Boucle sur chaque modèle pour lancer l'apprentissage (`train`) et mesurer le temps d'exécution.
#          5. **Test** : Évaluation des modèles sur le jeu de test (`test`), calcul des métriques et génération des matrices de confusion.
#          6. **Rapport** : Comparaison globale des performances via `Metrics.compare_models` et affichage d'un résumé console.
#
# @note Les résultats graphiques (courbes, matrices) sont sauvegardés dans le dossier `_results`.
# @author Romain Brouard et Paul Henry

import sys
import os
import time

from utils.DataHandler import DataHandler
from utils.Metrics import Metrics
from utils.functions import convert_time

from src.DecisionTreeClassifier import DecisionTreeClassifier
from src.RandomForestClassifier import RandomForestClassifier
from src.LogisticRegressionClassifier import LogisticRegressionClassifier
from src.SVMClassifier import SVMClassifier
from src.GBMClassifier import GBMClassifier
from src.CNNClassifier import CNNClassifier

# Configuration
t0 = time.time()
models_path = "_models"
results_path = "_results"
os.makedirs(models_path, exist_ok=True)
os.makedirs(results_path, exist_ok=True)

dataset_path = "data/cifar-10-batches-py"
n_train = 42000
n_val = 8000

# Chargement des données
print("Loading Data")
try:
    data_handler = DataHandler(dataset_path)
    data = data_handler.load_data(normalize=True, flatten=True)
    print(f"Training data: {len(data['train_data'])} samples")
    print(f"Test data:     {len(data['test_data'])} samples")
    print("-" * 30)
except FileNotFoundError as e:
    print(f"Error: {e}")
    sys.exit(1)

# Instanciation des modèles
print("\nInstantiating Models")
models = {
    "Decision Tree": DecisionTreeClassifier(n_train, n_val),
    "Random Forest": RandomForestClassifier(n_train, n_val),
    "Logistic Regression": LogisticRegressionClassifier(n_train, n_val),
    "SVM": SVMClassifier(n_train, n_val),
    "XGBoost": GBMClassifier(n_train, n_val),
    "CNN": CNNClassifier(n_train, n_val),
}

all_metrics = {}
training_times = {}
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

# Entraînement
print("\nStarting Training Phase")

for name, model in models.items():
    print(f"\n>> Training {name}...")
    start_time = time.time()

    _, val_metrics = model.train(data, class_names=class_names)

    duration = time.time() - start_time
    training_times[name] = duration
    print(f"{name} trained in {convert_time(duration)}")

# Test
print("\nStarting Testing Phase")

for name, model in models.items():
    print(f"\n>> Testing {name}...")

    # Calcule des métriques et génération de la matrice de confusion
    test_metrics = model.test(data, class_names=class_names)
    all_metrics[name] = test_metrics

    # Sauvegarde de la matrice de confusion
    if test_metrics.get("confusion_matrix_fig"):
        save_name = f"confusion_matrix_{name.replace(' ', '_').lower()}.png"
        save_path = os.path.join(results_path, save_name)
        test_metrics["confusion_matrix_fig"].savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")

    print(f"Test Accuracy: {test_metrics.get('accuracy', 0):.4f}")

# Comparaison & Rapport
print("\nFinal Comparison")

# Génération du graphique de comparation
Metrics.compare_models(
    all_metrics, save_path=os.path.join(results_path, "model_comparison.png")
)

print("\nSummary Table")
print(f"{'Model':<20} | {'Train Time':<15} | {'Test Acc':<10}")
print("-" * 50)

for name in models.keys():
    t_time = convert_time(training_times[name])
    test_acc = all_metrics[name].get("accuracy", 0)
    print(f"{name:<20} | {t_time:<15} | {test_acc:.4f}")

print(f"\nTotal execution time: {convert_time(time.time() - t0)}")
