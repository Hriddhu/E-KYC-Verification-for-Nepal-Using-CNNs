from pathlib import Path
import cv2
import numpy as np
from insightface.app import FaceAnalysis


def load_face_model(model_name: str = "buffalo_l", device: str | None = None):
    providers = ["CPUExecutionProvider"]
    if device == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    model = FaceAnalysis(name=model_name, providers=providers)
    ctx_id = 0 if device == "cuda" else -1
    model.prepare(ctx_id=ctx_id, det_size=(640, 640))
    return model


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
        image, pad_y, pad_y, pad_x, pad_x, borderType=cv2.BORDER_REPLICATE
    )


def _generate_candidate_images(img: np.ndarray) -> list[np.ndarray]:
    h, w = img.shape[:2]
    longest_side = max(h, w)
    scale = 1.0 if longest_side >= 320 else 320.0 / float(longest_side)
    resized = img if scale == 1.0 else cv2.resize(
        img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
    )

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
        key = (candidate.shape[0], candidate.shape[1], round(float(candidate.mean()), 2))
        if key in seen_shapes:
            continue
        seen_shapes.add(key)
        deduped.append(candidate)
    return deduped


def get_embedding(image_path: str, model: FaceAnalysis) -> np.ndarray:

    img = _read_image(image_path)
    multiple_faces_detected = False

    for candidate in _generate_candidate_images(img):
        faces = model.get(candidate)
        if len(faces) == 1:
            return np.array(faces[0].normed_embedding, dtype=np.float32)
        if len(faces) > 1:
            multiple_faces_detected = True

    if multiple_faces_detected:
        raise ValueError(f"Multiple faces detected in image: {Path(image_path).name}")
    raise ValueError(f"No face detected in image: {Path(image_path).name}")


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    if emb1.ndim == 1:
        emb1 = emb1.reshape(1, -1)
    if emb2.ndim == 1:
        emb2 = emb2.reshape(1, -1)
    return float(np.dot(emb1, emb2.T)[0][0])


def is_match(similarity: float, threshold: float) -> bool:
    return bool(similarity >= threshold)


def predict_face_similarity(
    doc_face_crop_path: str,
    selfie_path: str,
    model: FaceAnalysis,
    threshold: float,
) -> dict:
    """Called on EVERY request — model already loaded, just inference."""
    emb_doc = get_embedding(doc_face_crop_path, model)
    emb_selfie = get_embedding(selfie_path, model)

    similarity = cosine_similarity(emb_doc, emb_selfie)
    match = is_match(similarity, threshold)

    return {
        "similarity": similarity,
        "match": match,
        "threshold_used": float(threshold),
    }