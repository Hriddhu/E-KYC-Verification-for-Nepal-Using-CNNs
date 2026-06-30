from pathlib import Path
from typing import Any

import cv2
import numpy as np
import easyocr

from ml.docservice.detect_crop import DocumentDetector
from ml.ocr_service.parse_fields import parse_fields  # use parse_fields directly, not parse_back


def load_document_detector(front_model_path: str, back_model_path: str) -> DocumentDetector:
    return DocumentDetector(
        front_model_path=front_model_path,
        back_model_path=back_model_path,
    )


def load_ocr_reader(languages: list[str], gpu: bool = False) -> easyocr.Reader:
    return easyocr.Reader(languages, gpu=gpu)


def detect_front(detector: DocumentDetector, image_path: str, output_dir: str) -> list[dict[str, Any]]:
    return detector.detect_front(image_path=image_path, output_dir=output_dir)


def detect_back(detector: DocumentDetector, image_path: str, output_dir: str) -> list[dict[str, Any]]:
    return detector.detect_back(image_path=image_path, output_dir=output_dir)


def extract_back_fields(back_detections: list[dict[str, Any]]) -> dict[str, Any]:
    return parse_fields(back_detections)


def pick_photo_crop(detections: list[dict[str, Any]]) -> str | None:
    for detection in detections:
        class_name = str(detection.get("class_name", "")).lower()
        crop_path = detection.get("crop_path")
        if class_name == "photo" and crop_path:
            return str(crop_path)

    for detection in detections:
        crop_path = str(detection.get("crop_path") or "")
        if "photo" in Path(crop_path).stem.lower():
            return crop_path

    return None


def pick_logo_crop(detections: list[dict[str, Any]]) -> str | None:
    for detection in detections:
        class_name = str(detection.get("class_name", "")).lower()
        crop_path = detection.get("crop_path")
        if class_name in {"logo", "stamp", "government-stamp", "government_logo"} and crop_path:
            return str(crop_path)

    for detection in detections:
        crop_path = str(detection.get("crop_path") or "")
        stem = Path(crop_path).stem.lower()
        if "logo" in stem or "stamp" in stem:
            return crop_path

    return None


def _preprocess_stamp(image_path: str, size: tuple[int, int] = (256, 256)) -> np.ndarray:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unable to read stamp image: {image_path}")
    resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return cv2.equalizeHist(resized)


def compute_stamp_similarity(
    detected_stamp_path: str,
    standard_stamp_dir: Path,
    threshold: float,
) -> dict[str, Any]:
    """Called on EVERY request. Pure OpenCV, already fast — no preloading needed."""
    if not Path(detected_stamp_path).exists():
        raise ValueError(f"Detected stamp path does not exist: {detected_stamp_path}")

    template_paths = sorted(
        p for p in standard_stamp_dir.glob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    )
    if not template_paths:
        raise ValueError(f"No standard stamp templates found in: {standard_stamp_dir}")

    detected_img = _preprocess_stamp(detected_stamp_path)
    orb = cv2.ORB_create(nfeatures=500)
    detected_kp, detected_desc = orb.detectAndCompute(detected_img, None)

    best_match: dict[str, Any] | None = None

    for template_path in template_paths:
        template_img = _preprocess_stamp(str(template_path))
        template_kp, template_desc = orb.detectAndCompute(template_img, None)

        orb_score = 0.0
        if (
            detected_desc is not None
            and template_desc is not None
            and len(detected_kp) >= 10
            and len(template_kp) >= 10
        ):
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = matcher.match(detected_desc, template_desc)
            if matches:
                matches = sorted(matches, key=lambda m: m.distance)
                top_matches = matches[: max(10, int(len(matches) * 0.4))]
                avg_distance = sum(m.distance for m in top_matches) / len(top_matches)
                orb_score = float(np.clip(1.0 - (avg_distance / 100.0), 0.0, 1.0))

        template_score = float(
            cv2.matchTemplate(detected_img, template_img, cv2.TM_CCOEFF_NORMED)[0][0]
        )
        template_score = float(np.clip((template_score + 1) / 2, 0.0, 1.0))

        detected_hist = cv2.calcHist([detected_img], [0], None, [64], [0, 256])
        template_hist = cv2.calcHist([template_img], [0], None, [64], [0, 256])
        cv2.normalize(detected_hist, detected_hist)
        cv2.normalize(template_hist, template_hist)
        hist_score = float(cv2.compareHist(detected_hist, template_hist, cv2.HISTCMP_CORREL))
        hist_score = float(np.clip((hist_score + 1) / 2, 0.0, 1.0))

        final_score = round((0.5 * orb_score) + (0.3 * template_score) + (0.2 * hist_score), 4)

        candidate = {
            "standard_stamp_path": str(template_path),
            "score": final_score,
            "orb_score": round(orb_score, 4),
            "template_score": round(template_score, 4),
            "hist_score": round(hist_score, 4),
        }

        if best_match is None or candidate["score"] > best_match["score"]:
            best_match = candidate

    assert best_match is not None
    return {
        "detected_stamp_path": detected_stamp_path,
        "best_match": best_match,
        "threshold_used": threshold,
        "is_same_stamp": bool(best_match["score"] >= threshold),
    }