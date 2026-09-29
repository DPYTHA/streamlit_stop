from pathlib import Path
import json

import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

HISTORY_PATH = (
    PROJECT_ROOT
    / "results"
    / "training_history.json"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

# ============================================================
# CHARGEMENT
# ============================================================

with open(
    HISTORY_PATH,
    "r",
    encoding="utf-8"
) as f:
    history = json.load(f)


epochs = range(
    1,
    len(history["train_loss"]) + 1
)


# ============================================================
# LOSS
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    history["train_loss"],
    marker="o",
    label="Train Loss"
)

plt.plot(
    epochs,
    history["val_loss"],
    marker="o",
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "Training and Validation Loss"
)

plt.legend()
plt.grid(True)

plt.tight_layout()

loss_path = (
    RESULTS_DIR
    / "loss_curve.png"
)

plt.savefig(
    loss_path,
    dpi=300
)

plt.close()


# ============================================================
# ACCURACY
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    history["train_accuracy"],
    marker="o",
    label="Train Accuracy"
)

plt.plot(
    epochs,
    history["val_accuracy"],
    marker="o",
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title(
    "Training and Validation Accuracy"
)

plt.legend()
plt.grid(True)

plt.tight_layout()

accuracy_path = (
    RESULTS_DIR
    / "accuracy_curve.png"
)

plt.savefig(
    accuracy_path,
    dpi=300
)

plt.close()


# ============================================================
# FIN
# ============================================================

print("=" * 60)
print("TRAINING CURVES GENERATED")
print("=" * 60)

print()
print("Loss curve :")
print(loss_path)

print()
print("Accuracy curve :")
print(accuracy_path)