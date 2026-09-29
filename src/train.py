from pathlib import Path
import json
import copy

import torch
import torch.nn as nn
import torch.optim as optim

from Streamlit_ai.src.dataset import create_dataloaders
from Streamlit_ai.src.model import create_model


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

NUM_CLASSES = 2

# Dataset très petit :
# on commence avec un nombre raisonnable d'époques.
NUM_EPOCHS = 15

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

PATIENCE = 5


# ============================================================
# ENTRAÎNEMENT D'UNE ÉPOQUE
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        # Réinitialisation des gradients
        optimizer.zero_grad()

        # Forward
        outputs = model(images)

        # Loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Mise à jour des poids
        optimizer.step()

        # Statistiques
        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ============================================================
# VALIDATION
# ============================================================

def validate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            running_loss += (
                loss.item() * images.size(0)
            )

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STOP SIGN CLASSIFICATION - TRAINING")
    print("=" * 70)

    print()
    print("Device :", DEVICE)

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader
    ) = create_dataloaders()

    print()
    print("Classes :", train_dataset.classes)
    print("Mapping :", train_dataset.class_to_idx)

    print()
    print("Train :", len(train_dataset))
    print("Validation :", len(val_dataset))
    print("Test :", len(test_dataset))

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = create_model(
        num_classes=NUM_CLASSES
    )

    model = model.to(DEVICE)

    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # --------------------------------------------------------
    # LEARNING RATE SCHEDULER
    # --------------------------------------------------------

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )

    # --------------------------------------------------------
    # TRACKING
    # --------------------------------------------------------

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": []
    }

    best_val_loss = float("inf")

    best_model_state = None

    epochs_without_improvement = 0

    # --------------------------------------------------------
    # TRAINING LOOP
    # --------------------------------------------------------

    for epoch in range(NUM_EPOCHS):

        print()
        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS}"
        )
        print("-" * 50)

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            DEVICE
        )

        val_loss, val_accuracy = validate(
            model,
            val_loader,
            criterion,
            DEVICE
        )

        scheduler.step(val_loss)

        # Sauvegarde historique
        history["train_loss"].append(
            train_loss
        )

        history["train_accuracy"].append(
            train_accuracy
        )

        history["val_loss"].append(
            val_loss
        )

        history["val_accuracy"].append(
            val_accuracy
        )

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Train Loss      : {train_loss:.4f}"
        )

        print(
            f"Train Accuracy  : {train_accuracy:.4f}"
        )

        print(
            f"Val Loss        : {val_loss:.4f}"
        )

        print(
            f"Val Accuracy    : {val_accuracy:.4f}"
        )

        print(
            f"Learning Rate   : {current_lr:.6f}"
        )

        # ----------------------------------------------------
        # BEST MODEL
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_model_state = copy.deepcopy(
                model.state_dict()
            )

            epochs_without_improvement = 0

            print("✓ Nouveau meilleur modèle")

        else:

            epochs_without_improvement += 1

            print(
                f"Pas d'amélioration "
                f"({epochs_without_improvement}/{PATIENCE})"
            )

        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if epochs_without_improvement >= PATIENCE:

            print()
            print(
                "Early stopping déclenché."
            )

            break

    # ========================================================
    # RESTAURATION DU MEILLEUR MODÈLE
    # ========================================================

    if best_model_state is not None:

        model.load_state_dict(
            best_model_state
        )

    # ========================================================
    # SAUVEGARDE DU MODÈLE
    # ========================================================

    model_path = MODEL_DIR / "model.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_to_idx": train_dataset.class_to_idx,
            "classes": train_dataset.classes,
            "image_size": 224
        },
        model_path
    )

    # ========================================================
    # SAUVEGARDE DE L'HISTORIQUE
    # ========================================================

    history_path = RESULTS_DIR / "training_history.json"

    with open(
        history_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            history,
            f,
            indent=4
        )

    # ========================================================
    # FIN
    # ========================================================

    print()
    print("=" * 70)
    print("TRAINING TERMINÉ")
    print("=" * 70)

    print()
    print("Meilleur Val Loss :", best_val_loss)

    print()
    print("Modèle sauvegardé :")
    print(model_path)

    print()
    print("Historique sauvegardé :")
    print(history_path)


if __name__ == "__main__":
    main()