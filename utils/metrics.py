import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
)

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


def calculate_metrics(y_true, y_pred, model_name="Model"):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    print("\n" + "=" * 70)
    print(f"METRICS FOR: {model_name}")
    print("=" * 70)

    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    f1_per_class = f1_score(y_true, y_pred, average=None)
    precision_macro = precision_score(y_true, y_pred, average="macro")
    recall_macro = recall_score(y_true, y_pred, average="macro")

    print(f"\nOVERALL METRICS:")
    print(f"  Accuracy:           {accuracy:.4f}")
    print(f"  F1 Score (Macro):   {f1_macro:.4f}")
    print(f"  F1 Score (Weighted): {f1_weighted:.4f}")
    print(f"  Precision (Macro):  {precision_macro:.4f}")
    print(f"  Recall (Macro):     {recall_macro:.4f}")

    print(f"\nF1 SCORE PER CLASS:")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name:12s}: {f1_per_class[i]:.4f}")

    print(f"\nCLASSIFICATION REPORT:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    metrics = {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "f1_per_class": f1_per_class,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }

    return metrics


def plot_confusion_matrix(y_true, y_pred, model_name="Model", save_path=None):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Count"},
    )
    plt.title(f"Confusion Matrix - {model_name}", fontsize=16, fontweight="bold")
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Confusion matrix saved to: {save_path}")

    plt.show()

    return cm


def plot_training_history(history, model_name="Model", save_path=None):
    if hasattr(history, "history"):
        history = history.history

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history["accuracy"], label="Train Accuracy")
    ax1.plot(history["val_accuracy"], label="Val Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.set_title(f"{model_name} - Accuracy")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history["loss"], label="Train Loss")
    ax2.plot(history["val_loss"], label="Val Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title(f"{model_name} - Loss")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Training history saved to: {save_path}")

    plt.show()


def compare_models(metrics_dict, save_path="model_comparison.png"):
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print(
        f"\n{'Model':<20} {'Accuracy':<12} {'F1 (Macro)':<12} {'F1 (Weighted)':<12} {'Precision':<12} {'Recall':<12}"
    )
    print("-" * 90)

    for model_name, metrics in metrics_dict.items():
        print(
            f"{model_name:<20} {metrics['accuracy']:<12.4f} "
            f"{metrics['f1_macro']:<12.4f} {metrics['f1_weighted']:<12.4f} "
            f"{metrics['precision_macro']:<12.4f} {metrics['recall_macro']:<12.4f}"
        )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    models = list(metrics_dict.keys())
    accuracies = [metrics_dict[m]["accuracy"] for m in models]
    f1_macros = [metrics_dict[m]["f1_macro"] for m in models]
    f1_weighteds = [metrics_dict[m]["f1_weighted"] for m in models]

    axes[0].bar(models, accuracies, color="steelblue")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy Comparison")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(models, f1_macros, color="coral")
    axes[1].set_ylabel("F1 Score (Macro)")
    axes[1].set_title("F1 Macro Comparison")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(axis="y", alpha=0.3)

    axes[2].bar(models, f1_weighteds, color="seagreen")
    axes[2].set_ylabel("F1 Score (Weighted)")
    axes[2].set_title("F1 Weighted Comparison")
    axes[2].tick_params(axis="x", rotation=45)
    axes[2].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\nComparison chart saved to: {save_path}")
    plt.show()


def get_precision_recall_f1(y_true, y_pred, model_name="Model", average="macro"):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    if average is None:
        precision = precision_score(y_true, y_pred, average=None)
        recall = recall_score(y_true, y_pred, average=None)
        f1 = f1_score(y_true, y_pred, average=None)

        print(f"\n{model_name} - Per-Class Metrics:")
        print(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
        print("-" * 50)
        for i, class_name in enumerate(class_names):
            print(
                f"{class_name:<12} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f}"
            )

        return {
            "precision_per_class": precision,
            "recall_per_class": recall,
            "f1_per_class": f1,
        }
    else:
        precision = precision_score(y_true, y_pred, average=average)
        recall = recall_score(y_true, y_pred, average=average)
        f1 = f1_score(y_true, y_pred, average=average)

        print(f"\n{model_name} - Overall Metrics ({average}):")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")

        return {"precision": precision, "recall": recall, "f1_score": f1}


def get_predictions(model, data, model_type="sklearn"):
    if model_type == "pytorch":
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()

        if len(data.shape) == 2:
            data = data.reshape(-1, 32, 32, 3)
        data_tensor = torch.FloatTensor(data).permute(0, 3, 1, 2).to(device)

        with torch.no_grad():
            outputs = model(data_tensor)
            predictions = outputs.argmax(1).cpu().numpy()

        return predictions

    elif model_type == "keras":
        if len(data.shape) == 2:
            data = data.reshape(-1, 32, 32, 3)
        predictions = model.predict(data, verbose=0)
        predictions = np.argmax(predictions, axis=1)
    else:
        if len(data.shape) > 2:
            data = data.reshape(data.shape[0], -1)
        predictions = model.predict(data)

    return predictions
