import json
import shutil
from pathlib import Path
from collections import Counter

from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

SOURCE_DIR = Path(
    r"C:\Users\AGOUA MOUA\Documents\disc D\Computer vision cours\final-project-stop-signs-1-2025-04-25-t-06-47-41-058-z\final-project-stop-signs-1-2025-04-25-t-06-47-41-058-z"
)

ANNOTATIONS_FILE = SOURCE_DIR / "_annotations.json"

OUTPUT_DIR = Path("data/processed")

RANDOM_STATE = 42

TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15


# ============================================================
# VERIFICATION
# ============================================================

print("=" * 60)
print("STOP SIGN CLASSIFICATION - DATASET PREPARATION")
print("=" * 60)

if not SOURCE_DIR.exists():
    raise FileNotFoundError(
        f"\nDossier dataset introuvable :\n{SOURCE_DIR.resolve()}"
    )

if not ANNOTATIONS_FILE.exists():
    raise FileNotFoundError(
        f"\nFichier annotations introuvable :\n"
        f"{ANNOTATIONS_FILE.resolve()}"
    )

print(f"\nDataset : {SOURCE_DIR}")
print(f"Annotations : {ANNOTATIONS_FILE}")


# ============================================================
# CHARGEMENT DES ANNOTATIONS
# ============================================================

with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

print("\nType du dataset :", data.get("type"))
print("Labels :", data.get("labels"))

annotations = data.get("annotations", {})

if not isinstance(annotations, dict):
    raise ValueError(
        "Format inattendu : 'annotations' doit être un dictionnaire."
    )


# ============================================================
# CONSTRUCTION DE LA LISTE DES IMAGES
# ============================================================

images = []
missing_files = []
invalid_annotations = []

for filename, annotation_list in annotations.items():

    image_path = SOURCE_DIR / filename

    # Vérification de l'existence de l'image
    if not image_path.exists():
        missing_files.append(filename)
        continue

    # Chaque image possède une liste d'annotations
    if not isinstance(annotation_list, list) or len(annotation_list) == 0:
        invalid_annotations.append(filename)
        continue

    annotation = annotation_list[0]

    if not isinstance(annotation, dict):
        invalid_annotations.append(filename)
        continue

    label = annotation.get("label")

    if not label:
        invalid_annotations.append(filename)
        continue

    images.append(
        {
            "filename": filename,
            "label": label,
            "path": image_path,
        }
    )


# ============================================================
# RAPPORT
# ============================================================

print("\n" + "=" * 60)
print("VERIFICATION DU DATASET")
print("=" * 60)

print(f"Annotations trouvées : {len(annotations)}")
print(f"Images disponibles   : {len(images)}")
print(f"Images manquantes    : {len(missing_files)}")
print(f"Annotations invalides: {len(invalid_annotations)}")


# ============================================================
# ERREURS
# ============================================================

if missing_files:

    print("\nImages manquantes :")

    for filename in missing_files:
        print("  -", filename)

    raise FileNotFoundError(
        "\nCertaines images annotées sont introuvables."
    )


if invalid_annotations:

    print("\nAnnotations invalides :")

    for filename in invalid_annotations:
        print("  -", filename)

    raise ValueError(
        "\nCertaines annotations sont invalides."
    )


# ============================================================
# DISTRIBUTION DES CLASSES
# ============================================================

labels = [
    item["label"]
    for item in images
]

class_counts = Counter(labels)

print("\nDistribution des classes :")

for label, count in sorted(class_counts.items()):
    print(f"  {label:10s} : {count}")


# ============================================================
# VERIFICATION DES LABELS
# ============================================================

expected_labels = {
    "stop",
    "not_stop"
}

found_labels = set(labels)

unexpected_labels = found_labels - expected_labels

if unexpected_labels:

    raise ValueError(
        f"\nLabels inattendus détectés : {unexpected_labels}\n"
        f"Labels attendus : {expected_labels}"
    )


# ============================================================
# SPLIT TRAIN / TEMP
# ============================================================

train_data, temp_data = train_test_split(
    images,
    test_size=(VAL_SIZE + TEST_SIZE),
    random_state=RANDOM_STATE,
    stratify=labels
)


# ============================================================
# SPLIT VALIDATION / TEST
# ============================================================

temp_labels = [
    item["label"]
    for item in temp_data
]

val_data, test_data = train_test_split(
    temp_data,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=temp_labels
)


# ============================================================
# FONCTION DE COPIE
# ============================================================

def copy_dataset(split_name, dataset):

    split_dir = OUTPUT_DIR / split_name

    for item in dataset:

        label = item["label"]

        label_dir = split_dir / label

        label_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        destination = label_dir / item["filename"]

        shutil.copy2(
            item["path"],
            destination
        )


# ============================================================
# NETTOYAGE DU DOSSIER OUTPUT
# ============================================================

if OUTPUT_DIR.exists():

    print("\nNettoyage de l'ancien dataset processed...")

    for child in OUTPUT_DIR.iterdir():

        if child.is_dir():
            shutil.rmtree(child)

        else:
            child.unlink()


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CREATION DES DATASETS
# ============================================================

print("\nCréation des datasets...")

copy_dataset(
    "train",
    train_data
)

copy_dataset(
    "val",
    val_data
)

copy_dataset(
    "test",
    test_data
)


# ============================================================
# STATISTIQUES
# ============================================================

def count_split(dataset):

    return Counter(
        item["label"]
        for item in dataset
    )


print("\n" + "=" * 60)
print("DATASET FINAL")
print("=" * 60)

print(
    f"\nTrain : {len(train_data)} images"
)

print(
    dict(count_split(train_data))
)

print(
    f"\nValidation : {len(val_data)} images"
)

print(
    dict(count_split(val_data))
)

print(
    f"\nTest : {len(test_data)} images"
)

print(
    dict(count_split(test_data))
)


# ============================================================
# MANIFEST
# ============================================================

manifest = []

for split_name, dataset in [
    ("train", train_data),
    ("val", val_data),
    ("test", test_data),
]:

    for item in dataset:

        manifest.append(
            {
                "split": split_name,
                "filename": item["filename"],
                "label": item["label"],
            }
        )


manifest_file = OUTPUT_DIR / "manifest.json"

with open(
    manifest_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        manifest,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# FIN
# ============================================================

print("\n" + "=" * 60)
print("DATASET PREPARE AVEC SUCCES")
print("=" * 60)

print("\nDataset disponible dans :")
print(OUTPUT_DIR.resolve())

print(
    """
Structure :

data/
└── processed/
    ├── train/
    │   ├── stop/
    │   └── not_stop/
    │
    ├── val/
    │   ├── stop/
    │   └── not_stop/
    │
    ├── test/
    │   ├── stop/
    │   └── not_stop/
    │
    └── manifest.json
"""
)

print(f"Random state : {RANDOM_STATE}")