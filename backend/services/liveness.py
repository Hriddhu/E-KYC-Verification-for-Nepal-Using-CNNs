import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import torchvision.transforms as transforms
from collections import OrderedDict

from backend.core.model_loader import load_weights

CLASS_NAMES = ["live", "spoof"]
COMMON_CHECKPOINT_KEYS = ("state_dict", "model_state_dict", "model", "net")
DROPOUT_RATE = 0.3
DENSE_UNITS = 128
IMG_SIZE = 224


class MobileNetV2Liveness(nn.Module):
    def __init__(self, dropout_rate: float = DROPOUT_RATE, dense_units: int = DENSE_UNITS):
        super().__init__()
        self.backbone = models.mobilenet_v2(weights=None)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, dense_units),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dense_units, 1),
        )

    def forward(self, x):
        return self.backbone(x)


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, OrderedDict):
        return checkpoint

    if isinstance(checkpoint, dict):
        for key in COMMON_CHECKPOINT_KEYS:
            candidate = checkpoint.get(key)
            if isinstance(candidate, OrderedDict):
                return candidate

    return None


def _clean_state_dict_keys(state_dict: OrderedDict) -> OrderedDict:
    cleaned = OrderedDict()
    for key, value in state_dict.items():
        new_key = key[len("module."):] if key.startswith("module.") else key
        cleaned[new_key] = value
    return cleaned


def load_liveness_model(model_path: str):
    checkpoint = load_weights(model_path, device="cpu")

    # if someone saved the whole module instead of a state_dict
    if isinstance(checkpoint, nn.Module):
        checkpoint.eval()
        return checkpoint

    state_dict = _extract_state_dict(checkpoint)
    if state_dict is None:
        raise RuntimeError(
            f"Unsupported passive liveness model format: {type(checkpoint).__name__}"
        )

    state_dict = _clean_state_dict_keys(state_dict)
    model = MobileNetV2Liveness()

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        first_keys = list(state_dict.keys())[:20]
        raise RuntimeError(
            "Failed to load final_model.pt into MobileNetV2Liveness.\n"
            f"First checkpoint keys: {first_keys}\n"
            f"Original error: {exc}"
        ) from exc

    model.eval()
    return model


def preprocess_image(pil_img, img_size: int = IMG_SIZE):
    preprocess = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    return preprocess(pil_img).unsqueeze(0)


def predict_liveness(image_path: str, model, threshold: float) -> dict:
    pil_img = Image.open(image_path).convert("RGB")
    x = preprocess_image(pil_img)

    with torch.no_grad():
        output = model(x)

        if output.ndim == 2 and output.shape[1] == 1:
            spoof_probability = torch.sigmoid(output[0, 0]).item()
        elif output.ndim == 1 and output.shape[0] == 1:
            spoof_probability = torch.sigmoid(output[0]).item()
        else:
            probs = torch.softmax(output, dim=1)
            spoof_probability = probs[0, 1].item()

    spoof_probability = max(0.0, min(1.0, spoof_probability))
    pred_idx = 1 if spoof_probability >= threshold else 0
    pred_label = CLASS_NAMES[pred_idx]
    confidence = spoof_probability if pred_idx == 1 else (1.0 - spoof_probability)
    live_probability = 1.0 - spoof_probability

    return {
        "image_path": image_path,
        "prediction": pred_label,
        "passed": pred_label == "live",
        "spoof_probability": round(spoof_probability, 4),
        "live_probability": round(live_probability, 4),
        "confidence": round(confidence, 4),
        "threshold_used": float(threshold),
    }