from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis

app = None


def load_model():
    global app
    if app is None:
        app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        provider_names = {provider.lower() for provider in getattr(app, "providers", [])}
        ctx_id = 0 if any("cuda" in provider for provider in provider_names) else -1
        app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    return app


def _read_image(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    return img


def _expand_with_border(image: np.ndarray, border_ratio: float) -> np.ndarray:
    h, w = image.shape[:2]
    pad_y = max(24, int(h * border_ratio))
    pad_x = max(24, int(w * border_ratio))
    return cv2.copyMakeBorder(
        image,
        pad_y,
        pad_y,
        pad_x,
        pad_x,
        borderType=cv2.BORDER_REPLICATE,
    )


def _generate_candidate_images(img: np.ndarray) -> list[np.ndarray]:
    h, w = img.shape[:2]
    longest_side = max(h, w)
    scale = 1.0 if longest_side >= 320 else 320.0 / float(longest_side)
    resized = img if scale == 1.0 else cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    equalized = cv2.cvtColor(cv2.equalizeHist(gray), cv2.COLOR_GRAY2BGR)

    candidates = [
        resized,
        _expand_with_border(resized, 0.18),
        equalized,
        _expand_with_border(equalized, 0.22),
    ]

    deduped = []
    seen_shapes = set()
    for candidate in candidates:
        key = (candidate.shape[0], candidate.shape[1], candidate.mean().round(2))
        if key in seen_shapes:
            continue
        seen_shapes.add(key)
        deduped.append(candidate)
    return deduped


def get_embedding(image_path: str) -> np.ndarray:
    """
    Returns normalized 512-dim embedding from image.
    Retries with resized/padded variants for tightly cropped or low-resolution inputs.
    Raises ValueError if no face or multiple faces are found in all variants.
    """
    img = _read_image(image_path)

    app = load_model()
    multiple_faces_detected = False

    for candidate in _generate_candidate_images(img):
        faces = app.get(candidate)
        if len(faces) == 1:
            embedding = faces[0].normed_embedding
            return np.array(embedding, dtype=np.float32)
        if len(faces) > 1:
            multiple_faces_detected = True

    if multiple_faces_detected:
        raise ValueError(f"Multiple faces detected in image: {Path(image_path).name}")
    raise ValueError(f"No face detected in image: {Path(image_path).name}")
