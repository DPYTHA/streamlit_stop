import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


# ============================================================
# RESNET-18
# ============================================================

def create_model(num_classes=2):

    # Poids pré-entraînés ImageNet
    weights = ResNet18_Weights.DEFAULT

    model = resnet18(weights=weights)

    # Nombre d'entrées de la dernière couche
    in_features = model.fc.in_features

    # Remplacement de la couche de classification
    model.fc = nn.Linear(
        in_features,
        num_classes
    )

    return model


# ============================================================
# TEST DU MODÈLE
# ============================================================

if __name__ == "__main__":

    import torch

    print("=" * 60)
    print("STOP SIGN CLASSIFICATION - MODEL TEST")
    print("=" * 60)

    model = create_model(num_classes=2)

    print()
    print(model)

    # Test avec une image fictive
    x = torch.randn(1, 3, 224, 224)

    output = model(x)

    print()
    print("Input :", x.shape)
    print("Output :", output.shape)

    print()
    print("Model loaded successfully.")