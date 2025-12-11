import sys
import os
import time
import pickle
import torch

from utils.data_handler import load_all_data
from utils.functions import convert_time
from utils import metrics
from src import decision_tree
from src import random_forest
from src import logistic_regression
from src import svm
from src import gbm
from src import cnn
from src.cnn import CNNModel

t0 = time.time()

models_path = "_models"
os.makedirs(models_path, exist_ok=True)

results_path = "_results"
os.makedirs(results_path, exist_ok=True)

dataset_path = "data/cifar-10-batches-py"

class_names = [
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

n_train = 42000
n_val = 8000

print("Loading Data")
try:
    data = load_all_data(dataset_path)
    print(f"Training data: {len(data['train_data'])} samples")
    print(f"Test data: {len(data['test_data'])} samples")
    print(f"Feature dimension: {data['train_data'].shape[1]}")
    print()
except FileNotFoundError as e:
    print(f"Error: {e}")
    print(f"Check that path '{dataset_path}' contains CIFAR-10 files")
    sys.exit(1)

# Train Decision Tree
print("\nTraining decision tree...")
dtree_model, dtree_val_score = decision_tree.train(data, n_train, n_val)
t1 = time.time()
print(f"decision tree training time : {convert_time(t1 - t0)}")

print("\nTraining random forest...")
rf_model, rf_val_score = random_forest.train(data, n_train, n_val)
t2 = time.time()
print(f"random_forest training time : {convert_time(t2 - t1)}")

print("\nTraining logistic regression...")
lr_model, lr_val_score = logistic_regression.train(data, n_train, n_val)
t4_5 = time.time()
print(f"logistic regression training time : {convert_time(t4_5 - t2)}")

print("\nTraining svm...")
svm_model, svm_val_score = svm.train(data, n_train, n_val)
t3 = time.time()
print(f"svm training time : {convert_time(t3 - t4_5)}")

print("\nTraining: xgboost")
gbm_model, gbm_val_score = gbm.train(data, n_train, n_val)
t5 = time.time()
print(f"xgboost training time : {convert_time(t5 - t4)}")

print("\nTraining cnn...")
cnn_model, cnn_val_score = cnn.train(data, n_train, n_val)
t6 = time.time()
print(f"cnn training time : {convert_time(t6 - t5)}")

print("\nTest")
dtree_test_score = decision_tree.test(data)
rf_test_score = random_forest.test(data)
lr_test_score = logistic_regression.test(data)
svm_test_score = svm.test(data)
gbm_test_score = gbm.test(data)
cnn_test_score, cnn_test_loss = cnn.test(data)

print(f"\nResults")
print(f"Decision Tree       - Test Score: {dtree_test_score:.4f}")
print(f"Random Forest       - Test score: {rf_test_score:.4f}")
print(f"Logistic Regression - Test score: {lr_test_score:.4f}")
print(f"SVM                 - Test score: {svm_test_score:.4f}")
print(f"XGBoost             - Test score: {gbm_test_score:.4f}")
print(
    f"CNN                 - Test score: {cnn_test_score:.4f}   Test loss: {cnn_test_loss:.4f}"
)
print(f"Total training time: {convert_time(t6 - t0)}")

print("\nMetrics")

X_test = data["test_data"]
y_test = data["test_labels"]

all_metrics = {}

print("\nLoading Decision Tree model...")
with open("_models/DecisionTree.pkl", "rb") as f:
    model = pickle.load(f)
dt_preds = metrics.get_predictions(model, X_test, "sklearn")
dt_metrics = metrics.calculate_metrics(y_test, dt_preds, "Decision Tree")
metrics.plot_confusion_matrix(
    y_test,
    dt_preds,
    "Decision Tree",
    os.path.join(results_path, "confusion_matrix_dt.png"),
)
all_metrics["Decision Tree"] = dt_metrics

print("\nLoading Random Forest model...")
with open("_models/RandomForest.pkl", "rb") as f:
    model = pickle.load(f)
rf_preds = metrics.get_predictions(model, X_test, "sklearn")
rf_metrics = metrics.calculate_metrics(y_test, rf_preds, "Random Forest")
metrics.plot_confusion_matrix(
    y_test,
    rf_preds,
    "Random Forest",
    os.path.join(results_path, "confusion_matrix_rf.png"),
)
all_metrics["Random Forest"] = rf_metrics

print("\nLoading Logistic Regression model...")
with open("_models/LogisticRegression.pkl", "rb") as f:
    model = pickle.load(f)
lr_preds = metrics.get_predictions(model, X_test, "sklearn")
lr_metrics = metrics.calculate_metrics(y_test, lr_preds, "Logistic Regression")
metrics.plot_confusion_matrix(
    y_test,
    lr_preds,
    "Logistic Regression",
    os.path.join(results_path, "confusion_matrix_lr.png"),
)
all_metrics["Logistic Regression"] = lr_metrics

print("\nLoading RBF_SVM model...")
with open("_models/SVM.pkl", "rb") as f:
    model = pickle.load(f)
rbf_svm_preds = metrics.get_predictions(model, X_test, "sklearn")
rbf_svm_metrics = metrics.calculate_metrics(y_test, rbf_svm_preds, "SVM")
metrics.plot_confusion_matrix(
    y_test,
    rbf_svm_preds,
    "RBF_SVM",
    os.path.join(results_path, "confusion_matrix_rbf.png"),
)
all_metrics["SVM"] = rbf_svm_metrics

print("\nLoading GBM_Classifier model...")
with open("_models/GBM_Classifier.pkl", "rb") as f:
    model = pickle.load(f)
xgb_preds = metrics.get_predictions(model, X_test, "sklearn")
xgb_metrics = metrics.calculate_metrics(y_test, xgb_preds, "GBM_Classifier")
metrics.plot_confusion_matrix(
    y_test,
    xgb_preds,
    "GBM_Classifier",
    os.path.join(results_path, "confusion_matrix_xgb.png"),
)
all_metrics["XGBoost"] = xgb_metrics

print("\nLoading CNN model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel().to(device)
model.load_state_dict(torch.load("_models/CNN.pth", map_location=device))
model.eval()
cnn_preds = metrics.get_predictions(model, X_test, "pytorch")
cnn_metrics = metrics.calculate_metrics(y_test, cnn_preds, "CNN")
metrics.plot_confusion_matrix(
    y_test, cnn_preds, "CNN", os.path.join(results_path, "confusion_matrix_cnn.png")
)
all_metrics["CNN"] = cnn_metrics

print("COMPARING ALL MODELS")
metrics.compare_models(
    all_metrics, save_path=os.path.join(results_path, "model_comparison.png")
)

print(f"Total execution time: {convert_time(time.time() - t0)}")
print("\nResults saved in 'results/' directory:")
print("  - confusion_matrix_*.png for each model")
print("  - model_comparison.png for overall comparison")

print("\nPrecision, recall, F1-score per class")

metrics.get_precision_recall_f1(y_test, dt_preds, "Decision Tree", average=None)
metrics.get_precision_recall_f1(y_test, rf_preds, "Random Forest", average=None)
metrics.get_precision_recall_f1(y_test, lr_preds, "Logistic Regression", average=None)
metrics.get_precision_recall_f1(y_test, rbf_svm_preds, "SVM", average=None)
metrics.get_precision_recall_f1(y_test, xgb_preds, "XGBoost", average=None)
metrics.get_precision_recall_f1(y_test, cnn_preds, "CNN", average=None)

print(f"Total execution time: {convert_time(time.time() - t0)}")
print("\nResults saved in '_results/' directory:")
print("  - confusion_matrix_*.png for each model")
print("  - model_comparison.png for overall comparison")
