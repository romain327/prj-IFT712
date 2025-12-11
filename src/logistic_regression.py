import os
import pickle
import time
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

model_path = "_models/LogisticRegression.pkl"


def train(data_dict, n_train, n_val):
    # Convertir les données en float32
    X = np.array(data_dict["train_data"], dtype=np.float32)
    y = np.array(data_dict["train_labels"])

    # Utiliser uniquement les premiers (n_train + n_val) échantillons
    X_sub = X[: n_train + n_val]
    y_sub = y[: n_train + n_val]

    # Séparation stratifiée pour maintenir les proportions de classes
    X_train, X_val, y_train, y_val = train_test_split(
        X_sub,
        y_sub,
        train_size=n_train,
        test_size=n_val,
        random_state=42,
        stratify=y_sub,
    )

    print("Logistic Regression Classifier")

    model = make_pipeline(
        StandardScaler(),
        PCA(n_components=0.95),
        LogisticRegression(
            solver="lbfgs", max_iter=1000, C=1.0, random_state=42, verbose=1, n_jobs=-1
        ),
    )

    # Validation croisée
    print("Exécution de la validation croisée...")
    start_cv = time.time()

    cv_scores = cross_val_score(model, X_train, y_train, cv=3, n_jobs=-1, verbose=1)

    cv_time = time.time() - start_cv
    print(
        f"Validation croisée terminée en {cv_time:.1f} secondes ({cv_time / 60:.1f} minutes)"
    )
    print(f"Scores de validation croisée: {cv_scores}")
    print(f"Score moyen: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

    # Entraînement final sur l'ensemble d'entraînement complet
    print("Entraînement du modèle...")
    start_train = time.time()

    model.fit(X_train, y_train)

    train_time = time.time() - start_train
    print(
        f"Entraînement terminé en {train_time:.1f} secondes ({train_time / 60:.1f} minutes)"
    )

    # Évaluation de la validation
    val_score = model.score(X_val, y_val)
    print(f"Score de validation: {val_score:.4f}")

    # Sauvegarde du modèle entraîné
    os.makedirs("_models", exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Modèle sauvegardé dans {model_path}")

    return model, val_score


def test(data_dict):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Fichier modèle {model_path} non trouvé. "
            "Assurez-vous d'avoir entraîné le modèle avant de le tester."
        )

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    X_test = np.array(data_dict["test_data"], dtype=np.float32)
    y_test = np.array(data_dict["test_labels"])

    test_score = model.score(X_test, y_test)

    return test_score


def classify(image):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Fichier modèle {model_path} non trouvé. "
            "Assurez-vous d'avoir entraîné le modèle avant d'appeler classify()."
        )

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    image = np.array(image, dtype=np.float32).reshape(1, -1)
    predicted_class = model.predict(image)[0]

    return predicted_class


def get_probabilities(image):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Fichier modèle {model_path} non trouvé. "
            "Assurez-vous d'avoir entraîné le modèle avant d'appeler get_probabilities()."
        )

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    image = np.array(image, dtype=np.float32).reshape(1, -1)
    probabilities = model.predict_proba(image)[0]

    return probabilities
