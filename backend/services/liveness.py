import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import torchvision.transforms as transforms
from collections import OrderedDict

from backend.core.model_loader import load_weights


class MobileNetV2Liveness(nn.Module):
    """Architecture — copy exactly from passiveliveness.py, unchanged."""
    def __init__(self, dropout_rate=0.3, dense_units=128):
        pass

    def forward(self, x):
        pass


def _extract_state_dict(checkpoint):
    """Helper — copy exactly from passiveliveness.py."""
    pass


def _clean_state_dict_keys(state_dict):
    """Helper — copy exactly from passiveliveness.py, strips 'module.' prefix."""
    pass


def load_liveness_model(model_path: str):
    """
    Called ONCE at startup.
    1. checkpoint = load_weights(model_path, device="cpu")   ← use the abstraction
    2. state_dict = _extract_state_dict(checkpoint)
    3. state_dict = _clean_state_dict_keys(state_dict)
    4. build MobileNetV2Liveness()
    5. model.load_state_dict(state_dict)
    6. model.eval()
    7. return model
    """
    pass


def preprocess_image(pil_img, img_size: int = 224):
    """Preprocessing — copy exactly from passiveliveness.py."""
    pass


def predict_liveness(image_path: str, model, threshold: float) -> dict:
    """
    Called on EVERY request — model already loaded, just inference.
    1. open image with PIL, convert to RGB
    2. preprocess_image()
    3. run model(tensor) inside torch.no_grad()
    4. sigmoid the output logit to get spoof_probability
    5. compute live_probability = 1 - spoof_probability
    6. compare against threshold for prediction
    7. return result dict
    """
    pass