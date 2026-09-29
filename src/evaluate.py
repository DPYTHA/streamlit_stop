from pathlib import Path
import json

import torch
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns

from Streamlit_ai.src.dataset import create_dataloaders
from Streamlit_ai.src.model import create_model


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "model.pt"

RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STOP SIGN CLASSIFICATION - TEST EVALUATION")
    print("=" * 70)

    print()
    print("Device :", DEVICE)

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader
    ) = create_dataloaders()

    class_names = test_dataset.classes

    print()
    print("Classes :", class_names)
    print("Test images :", len(test_dataset))

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = create_model(
        num_classes=len(class_names)
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(DEVICE)

    model.eval()

    print()
    print("Modèle chargé :", MODEL_PATH)

    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)

            outputs = model(images)

            probabilities = F.softmax(
                outputs,
                dim=1
            )

            predictions = outputs.argmax(
                dim=1
            )

            all_labels.extend(
                labels.numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        average="binary",
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        average="binary",
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        average="binary",
        zero_division=0
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(
        all_labels,
        all_predictions
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    report = classification_report(
        all_labels,
        all_predictions,
        target_names=class_names,
        zero_division=0
    )

    print()
    print("=" * 70)
    print("TEST RESULTS")
    print("=" * 70)

    print()
    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1-score  : {f1:.4f}"
    )

    print()
    print("Classification Report")
    print("-" * 70)
    print(report)

    print()
    print("Confusion Matrix")
    print(cm)

    # --------------------------------------------------------
    # SAVE METRICS
    # --------------------------------------------------------

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "test_samples": len(test_dataset),
        "classes": class_names,
        "confusion_matrix": cm.tolist()
    }

    metrics_path = RESULTS_DIR / "metrics.json"

    with open(
        metrics_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # CONFUSION MATRIX PLOT
    # --------------------------------------------------------

    plt.figure(
        figsize=(7, 6)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.xlabel(
        "Predicted label"
    )

    plt.ylabel(
        "True label"
    )

    plt.title(
        "Stop Sign Classification - Confusion Matrix"
    )

    plt.tight_layout()

    confusion_path = (
        RESULTS_DIR /
        "confusion_matrix.png"
    )

    plt.savefig(
        confusion_path,
        dpi=300
    )

    plt.close()

    # --------------------------------------------------------
    # FIN
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("EVALUATION TERMINÉE")
    print("=" * 70)

    print()
    print("Metrics :", metrics_path)

    print(
        "Confusion matrix :",
        confusion_path
    )


if __name__ == "__main__":
    main()