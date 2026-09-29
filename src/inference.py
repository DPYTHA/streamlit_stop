from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F
from PIL import Image

from torchvision import transforms
from src.model import create_model



# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "model.pt"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGE_SIZE = 224


# ============================================================
# TRANSFORMATION
# ============================================================

transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# CHARGEMENT DU MODÈLE
# ============================================================

def load_model():

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    classes = checkpoint["classes"]

    model = create_model(
        num_classes=len(classes)
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(DEVICE)

    model.eval()

    return model, classes


# ============================================================
# PRÉDICTION
# ============================================================

def predict_image(
    image_path,
    model,
    classes
):

    image_path = Path(image_path)

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image introuvable : {image_path}"
        )

    # Chargement de l'image
    image = Image.open(
        image_path
    ).convert("RGB")

    # Transformation
    image_tensor = transform(
        image
    )

    # Ajout dimension batch
    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(
        DEVICE
    )

    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    start_time = time.perf_counter()

    with torch.no_grad():

        outputs = model(
            image_tensor
        )

        probabilities = F.softmax(
            outputs,
            dim=1
        )

        predicted_index = (
            probabilities.argmax(
                dim=1
            ).item()
        )

        confidence = (
            probabilities[0][predicted_index]
            .item()
        )

    end_time = time.perf_counter()

    inference_time_ms = (
        end_time - start_time
    ) * 1000

    predicted_class = (
        classes[predicted_index]
    )

    return {
        "prediction": predicted_class,
        "confidence": confidence,
        "inference_time_ms": inference_time_ms
    }


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

def main():

    print("=" * 70)
    print("STOP SIGN CLASSIFICATION - IMAGE INFERENCE")
    print("=" * 70)

    print()
    print("Device :", DEVICE)

    # --------------------------------------------------------
    # VÉRIFICATION DE L'ARGUMENT
    # --------------------------------------------------------

    if len(sys.argv) < 2:

        print()
        print("Usage :")

        print(
            "python inference.py "
            "chemin_vers_image.jpg"
        )

        print()
        print("Exemple :")

        print(
            "python inference.py "
            "..\\data\\processed\\test\\stop\\image.jpg"
        )

        return

    image_path = sys.argv[1]

    # --------------------------------------------------------
    # MODÈLE
    # --------------------------------------------------------

    model, classes = load_model()

    print()
    print("Classes :", classes)

    print(
        "Modèle :",
        MODEL_PATH
    )

    # --------------------------------------------------------
    # PRÉDICTION
    # --------------------------------------------------------

    result = predict_image(
        image_path,
        model,
        classes
    )

    # --------------------------------------------------------
    # RESULTAT
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PREDICTION")
    print("=" * 70)

    print()
    print("Image :")
    print(image_path)

    print()
    print(
        "Prediction :",
        result["prediction"]
    )

    print(
        "Confidence :",
        f"{result['confidence'] * 100:.2f}%"
    )

    print(
        "Inference time :",
        f"{result['inference_time_ms']:.2f} ms"
    )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()