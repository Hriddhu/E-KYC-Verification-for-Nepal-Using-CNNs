import torch
import torch.nn as nn
import timm
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from functools import lru_cache

from backend.core.model_loader import load_weights


class EfficientNetBinary(nn.Module):
    def __init__(self, model_name: str = "efficientnet_b0", pretrained: bool = False):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        in_features = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        feats = self.backbone(x)
        return self.classifier(feats)


@lru_cache(maxsize=2)
def get_transforms(img_size: int = 224):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def load_tampering_model(model_path: str, device: str | None = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = EfficientNetBinary(model_name="efficientnet_b0", pretrained=False)
    state = load_weights(model_path, device=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, device


def predict_tampering(image_path: str, model, device: str, threshold: float) -> dict:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    tensor = get_transforms()(image=image)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        tampered_probability = float(torch.sigmoid(logits).item())

    tampered_probability = max(0.0, min(1.0, tampered_probability))
    genuine_probability = 1.0 - tampered_probability
    is_tampered = tampered_probability >= threshold

    return {
        "image_path": image_path,
        "tampered_probability": round(tampered_probability, 4),
        "genuine_probability": round(genuine_probability, 4),
        "threshold_used": threshold,
        "is_tampered": is_tampered,
        "prediction": "FORGED" if is_tampered else "GENUINE",
        "device": device,
    }