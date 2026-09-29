from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"

IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 0  # Windows : 0 est plus sûr


# ============================================================
# TRANSFORMATIONS
# ============================================================

# Transformations utilisées uniquement pour l'entraînement.
# Elles permettent d'améliorer la généralisation du modèle.
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomRotation(
        degrees=10
    ),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.15
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Validation et test :
# aucune augmentation aléatoire.
eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# DATASETS
# ============================================================

def create_datasets():

    train_dir = DATA_DIR / "train"
    val_dir = DATA_DIR / "val"
    test_dir = DATA_DIR / "test"

    # Vérification des dossiers
    for directory in [train_dir, val_dir, test_dir]:
        if not directory.exists():
            raise FileNotFoundError(
                f"Dossier introuvable : {directory}"
            )

    train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        root=val_dir,
        transform=eval_transform
    )

    test_dataset = datasets.ImageFolder(
        root=test_dir,
        transform=eval_transform
    )

    return train_dataset, val_dataset, test_dataset


# ============================================================
# DATALOADERS
# ============================================================

def create_dataloaders():

    train_dataset, val_dataset, test_dataset = create_datasets()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader
    )


# ============================================================
# TEST DU DATASET
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("STOP SIGN CLASSIFICATION - DATASET TEST")
    print("=" * 60)

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

    print()
    print("Nombre de batches :")
    print("Train :", len(train_loader))
    print("Validation :", len(val_loader))
    print("Test :", len(test_loader))

    # Vérification d'un batch
    images, labels = next(iter(train_loader))

    print()
    print("Premier batch :")
    print("Images :", images.shape)
    print("Labels :", labels.shape)

    print()
    print("Dataset chargé avec succès.")