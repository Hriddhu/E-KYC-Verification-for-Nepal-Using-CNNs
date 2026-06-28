import torch
import torch.nn as nn
import timm
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from functools import lru_cache

from backend.core.model_loader import load_weights


class EfficientNetBinary(nn.Module):
    """Architecture — copy exactly from predict_baseline.py, unchanged."""
    def __init__(self, model_name="efficientnet_b0", pretrained=False):
        pass

    def forward(self, x):
        pass


@lru_cache(maxsize=2)
def get_transforms(img_size=224):
    """
    Preprocessing pipeline — copy exactly from predict_baseline.py.
    Safe to cache: this is a lightweight object, not a model.
    """
    pass


def load_tampering_model(model_path: str, device: str | None = None):
    """
    Called ONCE at startup.
    1. pick device (cuda if available else cpu)
    2. build EfficientNetBinary()
    3. state = load_weights(model_path, device)   ← use the abstraction, not torch.load directly
    4. model.load_state_dict(state)
    5. model.to(device)
    6. model.eval()
    7. return (model, device)
    """
    pass


def predict_tampering(image_path: str, model, device: str, threshold: float) -> dict:
    """
    Called on EVERY request — model already loaded, just inference.
    1. read image with cv2, convert BGR to RGB
    2. apply get_transforms(), move tensor to device
    3. run model(tensor) inside torch.no_grad()
    4. sigmoid the output to get tampered_probability
    5. compute genuine_probability = 1 - tampered_probability
    6. compare against threshold to decide is_tampered
    7. return result dict
    """
    pass