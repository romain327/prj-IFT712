import os
import pickle
import time
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, train_test_split

# Path where the trained model will be stored
model_path = "_models/DecisionTree.pkl"


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

    print("Decision Tree Classifier")

    model = DecisionTreeClassifier(
        criterion="gini",  # Gini impurity (can also use 'entropy')
        max_depth=30,  # Limit depth to prevent overfitting
        min_samples_split=20,  # Minimum samples required to split
        min_samples_leaf=10,  # Minimum samples in leaf nodes
        max_features="sqrt",  # Number of features to consider for split
        random_state=42,
    )

    # Validation croisée
    print("Running cross-validation...")
    start_cv = time.time()

    cv_scores = cross_val_score(model, X_train, y_train, cv=5, n_jobs=-1)

    cv_time = time.time() - start_cv
    print(
        f"Cross-validation completed in {cv_time:.1f} seconds ({cv_time / 60:.1f} minutes)"
    )
    print(f"Mean score: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

    # Entraînement final sur l'ensemble d'entraînement complet
    print("Training model...")
    start_train = time.time()

    model.fit(X_train, y_train)

    train_time = time.time() - start_train
    print(
        f"Training completed in {train_time:.1f} seconds "
        f"({train_time / 60:.1f} minutes)"
    )

    # Évaluation de la validation
    val_score = model.score(X_val, y_val)
    print(f"Validation score: {val_score:.4f}")

    # Feature importance (optional but interesting)
    feature_importance = model.feature_importances_
    top_features = np.argsort(feature_importance)[-10:][::-1]
    print(f"Top 10 most important features: {top_features}")

    # Sauvegarde du modèle entraîné
    os.makedirs("_models", exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved to {model_path}")

    return model, val_score


def test(data_dict):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file {model_path} not found. "
            "Make sure you have trained the model before testing."
        )

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Convert test data to a consistent dtype
    X_test = np.array(data_dict["test_data"], dtype=np.float32)
    y_test = np.array(data_dict["test_labels"])

    # Evaluate model
    test_score = model.score(X_test, y_test)

    return test_score


def classify(image):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file {model_path} not found. "
            "Make sure you have trained the model before calling classify()."
        )

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    image = np.array(image, dtype=np.float32).reshape(1, -1)

    predicted_class = model.predict(image)[0]

    return predicted_class


def get_probabilities(image):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file {model_path} not found. "
            "Make sure you have trained the model before calling get_probabilities()."
        )

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    image = np.array(image, dtype=np.float32).reshape(1, -1)

    probabilities = model.predict_proba(image)[0]

    return probabilities
