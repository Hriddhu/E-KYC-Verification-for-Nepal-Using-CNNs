import base64
import importlib.util
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Any
from uuid import uuid4

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

ROOT_DIR = Path(__file__).resolve().parents[3]
LIVENESS_SCRIPT = ROOT_DIR / "ml" / "liveness-service" / "liveness.py"
OUTPUT_DIR = ROOT_DIR / "ml" / "liveness-service" / "extracted_faces"
UPLOAD_DIR = ROOT_DIR / "ml" / "docservice" / "uploads"
CROP_OUTPUT_DIR = ROOT_DIR / "ml" / "docservice" / "crops"
STANDARD_STAMP_DIR = ROOT_DIR / "ml" / "docservice" / "standard_stamps"
PASSIVE_LIVENESS_SCRIPT = ROOT_DIR / "ml" / "liveness-service" / "passiveliveness.py"
PASSIVE_LIVENESS_MODEL_PATH = ROOT_DIR / "ml" / "liveness-service" / "final_model.pt"
TAMPERING_SCRIPT = ROOT_DIR / "ml" / "tampering" / "predict_baseline.py"
TAMPERING_MODEL_PATH = ROOT_DIR / "ml" / "tampering" / "best_model.pth"

DOC_SERVICE_DIR = ROOT_DIR / "ml" / "docservice"
if str(DOC_SERVICE_DIR) not in sys.path:
    sys.path.append(str(DOC_SERVICE_DIR))

OCR_SERVICE_SRC_DIR = ROOT_DIR / "ml" / "ocr-service" / "src"
if str(OCR_SERVICE_SRC_DIR) not in sys.path:
    sys.path.append(str(OCR_SERVICE_SRC_DIR))

FACE_SERVICE_SRC_DIR = ROOT_DIR / "ml" / "face-service" / "src"
if str(FACE_SERVICE_SRC_DIR) not in sys.path:
    sys.path.append(str(FACE_SERVICE_SRC_DIR))

from detect_crop import DocumentDetector
from parse_fields import parse_back

# ─────────────────────────────────────────────────────────────
# OPTIONAL FACE MATCH MODULES (DON'T BREAK API IF MISSING)
# ─────────────────────────────────────────────────────────────
try:
    from embed import get_embedding
    from match import compare_embeddings
except Exception as e:
    print(f"[FACE-MODULES] WARNING: embed/match not available -> {e}")
    get_embedding = None
    compare_embeddings = None

print("[FACE-MODULES] get_embedding:", "OK" if get_embedding else "MISSING")
print("[FACE-MODULES] compare_embeddings:", "OK" if compare_embeddings else "MISSING")

app = FastAPI(title="KYC Liveness API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


detector = DocumentDetector()

KYC_WEIGHTS = {
    "face_similarity": 0.3,
    "stamp_similarity": 0.10,
    "ocr_accuracy": 0.20,
    "passive_liveness": 0.2,
    "document_tampering": 0.2,
}
KYC_APPROVAL_THRESHOLD = 0.55
PASSIVE_LIVENESS_THRESHOLD = 0.5
TAMPERING_THRESHOLD = 0.78


def _save_upload(file: UploadFile, destination_dir: Path, prefix: str) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    extension = Path(file.filename or "").suffix or ".jpg"
    destination_path = destination_dir / f"{prefix}_{uuid4().hex}{extension}"

    with destination_path.open("wb") as out_file:
        out_file.write(file.file.read())

    return destination_path


def _pick_photo_crop(detections: list[dict[str, Any]]) -> str | None:
    """Pick the citizenship photo crop path from detector outputs."""
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


def _pick_logo_crop(detections: list[dict[str, Any]]) -> str | None:
    """Pick the government logo/stamp crop path from detector outputs."""
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


def _compute_stamp_similarity(
    detected_stamp_path: str,
    standard_stamp_dir: Path = STANDARD_STAMP_DIR,
    threshold: float = 0.72,
) -> dict[str, Any]:
    """Compare a detected government stamp crop against standard stamp templates."""
    if not Path(detected_stamp_path).exists():
        raise ValueError(f"Detected stamp path does not exist: {detected_stamp_path}")

    template_paths = sorted(
        [
            p
            for p in standard_stamp_dir.glob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        ]
    )
    if not template_paths:
        raise ValueError(
            "No standard stamp templates found. "
            f"Add template images to: {standard_stamp_dir}"
        )

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

        template_score = float(cv2.matchTemplate(detected_img, template_img, cv2.TM_CCOEFF_NORMED)[0][0])
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


def _run_face_similarity(doc_face_crop_path: str, selfie_path: str) -> dict[str, Any] | None:
    """
    Returns normalized keys:
      - similarity (float)
      - match (bool)
      - threshold_used (float)
    """
    if not get_embedding or not compare_embeddings:
        return None

    try:
        emb_doc = get_embedding(doc_face_crop_path)
    except ValueError as exc:
        raise ValueError(f"Document photo face detection failed: {exc}") from exc

    try:
        emb_selfie = get_embedding(selfie_path)
    except ValueError as exc:
        raise ValueError(f"Selfie face detection failed: {exc}") from exc

    if emb_doc is None or emb_selfie is None:
        raise ValueError("Face embedding could not be generated (no face detected or model failed).")

    similarity_result = compare_embeddings(emb_doc, emb_selfie)

    return {
        "similarity": float(similarity_result.get("similarity", 0.0)),
        "match": bool(similarity_result.get("match", False)),
        "threshold_used": float(similarity_result.get("threshold_used", 0.0)),
    }


def _normalize_string(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", value.lower())).strip()


def _date_from_form(date_of_birth: str) -> tuple[str, str, str]:
    try:
        dt = datetime.strptime(date_of_birth, "%Y-%m-%d")
        return str(dt.year), dt.strftime("%b").lower(), str(dt.day)
    except ValueError:
        return "", "", ""


def _date_from_ocr(ocr_dob: str) -> tuple[str, str, str]:
    year_match = re.search(r"Year:\s*(\d{4})", ocr_dob, re.IGNORECASE)
    month_match = re.search(r"Month:\s*([A-Za-z]{3})", ocr_dob, re.IGNORECASE)
    day_match = re.search(r"Day:\s*(\d{1,2})", ocr_dob, re.IGNORECASE)
    return (
        year_match.group(1) if year_match else "",
        month_match.group(1).lower() if month_match else "",
        day_match.group(1).lstrip("0") if day_match else "",
    )


def _score_similarity(expected: str, actual: str) -> float:
    if not actual or actual in {"[NOT_DETECTED]", "[PARTIAL]", "[INVALID]"}:
        return 0.1
    exp = _normalize_string(expected)
    act = _normalize_string(actual)
    if not exp or not act:
        return 0.1
    return round(max(0.1, SequenceMatcher(None, exp, act).ratio()), 3)


def _compute_field_accuracy(
    *,
    full_name: str,
    date_of_birth: str,
    citizenship_number: str,
    permanent_address: str,
    parsed_fields: dict[str, Any],
) -> dict[str, Any]:
    extracted_name = str(parsed_fields.get("full-name", {}).get("cleaned", ""))
    extracted_citizenship = str(parsed_fields.get("citisenship-number", {}).get("cleaned", ""))
    extracted_dob = str(parsed_fields.get("DOB", {}).get("cleaned", ""))
    extracted_address = str(parsed_fields.get("permanent-address", {}).get("cleaned", ""))

    name_accuracy = _score_similarity(full_name, extracted_name)
    citizenship_accuracy = _score_similarity(citizenship_number, extracted_citizenship)

    form_year, form_month, form_day = _date_from_form(date_of_birth)
    ocr_year, ocr_month, ocr_day = _date_from_ocr(extracted_dob)
    dob_parts = [
        1.0 if form_year and form_year == ocr_year else 0.0,
        1.0 if form_month and form_month == ocr_month else 0.0,
        1.0 if form_day and form_day == ocr_day else 0.0,
    ]
    dob_accuracy = round(max(0.1, sum(dob_parts) / 3), 3)
    if extracted_dob in {"", "[NOT_DETECTED]", "[PARTIAL]", "[INVALID]"}:
        dob_accuracy = 0.1

    address_accuracy = _score_similarity(permanent_address, extracted_address)
    if "wardNo" in permanent_address:
        ward_match = re.search(r'"wardNo"\s*:\s*"?(\d{1,2})', permanent_address)
        ocr_ward_match = re.search(r"Ward\s*No:\s*(\d{1,2})", extracted_address, re.IGNORECASE)
        if ward_match and ocr_ward_match:
            address_accuracy = round((address_accuracy + (1.0 if ward_match.group(1) == ocr_ward_match.group(1) else 0.1)) / 2, 3)

    field_accuracy = {
        "full_name": name_accuracy,
        "date_of_birth": dob_accuracy,
        "citizenship_number": citizenship_accuracy,
        "permanent_address": address_accuracy,
    }
    overall_accuracy = round(sum(field_accuracy.values()) / len(field_accuracy), 3)
    return {
        "field_accuracy": field_accuracy,
        "overall_accuracy": overall_accuracy,
        "extracted_values": {
            "full_name": extracted_name,
            "date_of_birth": extracted_dob,
            "citizenship_number": extracted_citizenship,
            "permanent_address": extracted_address,
        },
    }


def _safe_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, score))


def _print_terminal_result(label: str, payload: dict[str, Any]) -> None:
    print(f"[{label}-RESULT] {json.dumps(payload, indent=2, default=str)}")


@lru_cache(maxsize=4)
def _load_external_module(module_name: str, module_path: Path):
    if not module_path.exists():
        raise FileNotFoundError(f"Module not found: {module_path}")

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_passive_liveness(selfie_path: str) -> dict[str, Any]:
    module = _load_external_module("passive_liveness_module", PASSIVE_LIVENESS_SCRIPT)
    result = module.predict_image(
        image_path=selfie_path,
        model_path=str(PASSIVE_LIVENESS_MODEL_PATH),
        threshold=PASSIVE_LIVENESS_THRESHOLD,
    )
    return {
        "selfie_path": result["image_path"],
        "live_probability": _safe_score(result.get("live_probability")),
        "spoof_probability": _safe_score(result.get("spoof_probability")),
        "passed": bool(result.get("passed", False)),
        "threshold_used": float(result.get("threshold_used", PASSIVE_LIVENESS_THRESHOLD)),
        "prediction": str(result.get("prediction", "unknown")),
        "model_path": str(result.get("model_path", PASSIVE_LIVENESS_MODEL_PATH)),
    }


def _run_document_tampering(image_path: str, side: str) -> dict[str, Any]:
    module = _load_external_module("tampering_module", TAMPERING_SCRIPT)
    result = module.predict_image(
        image_path=image_path,
        model_path=str(TAMPERING_MODEL_PATH),
        threshold=TAMPERING_THRESHOLD,
    )
    return {
        "side": side,
        "image_path": str(result.get("image_path", image_path)),
        "genuine_probability": _safe_score(result.get("genuine_probability")),
        "tampered_probability": _safe_score(result.get("tampered_probability")),
        "is_tampered": bool(result.get("is_tampered", False)),
        "threshold_used": float(result.get("threshold_used", TAMPERING_THRESHOLD)),
        "prediction": str(result.get("prediction", "UNKNOWN")),
        "model_path": str(result.get("model_path", TAMPERING_MODEL_PATH)),
        "device": result.get("device"),
    }


def _compute_document_tampering(front_image_path: str, back_image_path: str) -> dict[str, Any]:
    front_result = _run_document_tampering(front_image_path, "front")
    back_result = _run_document_tampering(back_image_path, "back")

    overall_genuine_score = round(
        (front_result["genuine_probability"] + back_result["genuine_probability"]) / 2,
        4,
    )

    return {
        "front": front_result,
        "back": back_result,
        "overall_genuine_score": overall_genuine_score,
        "any_tampered": bool(front_result["is_tampered"] or back_result["is_tampered"]),
        "threshold_used": TAMPERING_THRESHOLD,
    }


def _compute_kyc_decision(
    accuracy_report: dict[str, Any],
    face_similarity: dict[str, Any] | None,
    stamp_similarity: dict[str, Any] | None,
    passive_liveness: dict[str, Any] | None,
    document_tampering: dict[str, Any] | None,
) -> dict[str, Any]:
    face_score = _safe_score((face_similarity or {}).get("similarity"))
    stamp_score = _safe_score(((stamp_similarity or {}).get("best_match") or {}).get("score"))
    ocr_score = _safe_score((accuracy_report or {}).get("overall_accuracy"))
    passive_score = _safe_score((passive_liveness or {}).get("live_probability"))
    tampering_score = _safe_score((document_tampering or {}).get("overall_genuine_score"))

    weighted_score = round(
        (face_score * KYC_WEIGHTS["face_similarity"])
        + (stamp_score * KYC_WEIGHTS["stamp_similarity"])
        + (ocr_score * KYC_WEIGHTS["ocr_accuracy"])
        + (passive_score * KYC_WEIGHTS["passive_liveness"])
        + (tampering_score * KYC_WEIGHTS["document_tampering"]),
        4,
    )

    approved = weighted_score >= KYC_APPROVAL_THRESHOLD
    status = "approved" if approved else "rejected"
    return {
        "status": status,
        "approved": approved,
        "overall_score": weighted_score,
        "threshold": KYC_APPROVAL_THRESHOLD,
        "weights": KYC_WEIGHTS,
        "component_scores": {
            "face_similarity": face_score,
            "stamp_similarity": stamp_score,
            "ocr_accuracy": ocr_score,
            "passive_liveness": passive_score,
            "document_tampering": tampering_score,
        },
        "decision_reason": (
            f"KYC {status.upper()} (weighted score {weighted_score:.4f} "
            f"{'≥' if approved else '<'} threshold {KYC_APPROVAL_THRESHOLD:.2f})"
        ),
    }


@app.post("/api/kyc/upload")
def upload_kyc_documents(
    full_name: str = Form(...),
    date_of_birth: str = Form(...),
    gender: str = Form(...),
    citizenship_number: str = Form(...),
    permanent_address: str = Form(...),
    current_address: str = Form(...),
    selfie_image: UploadFile = File(...),
    document_front: UploadFile = File(...),
    document_back: UploadFile = File(...),
):
    del gender, current_address

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    front_image_path = _save_upload(
        file=document_front,
        destination_dir=UPLOAD_DIR,
        prefix=f"front_{timestamp}",
    )
    back_image_path = _save_upload(
        file=document_back,
        destination_dir=UPLOAD_DIR,
        prefix=f"back_{timestamp}",
    )
    selfie_path = _save_upload(
        file=selfie_image,
        destination_dir=UPLOAD_DIR,
        prefix=f"selfie_{timestamp}",
    )

    front_crop_dir = CROP_OUTPUT_DIR / timestamp / "front"
    back_crop_dir = CROP_OUTPUT_DIR / timestamp / "back"

    front_detections = detector.detect_front(
        image_path=str(front_image_path),
        output_dir=str(front_crop_dir),
    )
    back_detections = detector.detect_back(
        image_path=str(back_image_path),
        output_dir=str(back_crop_dir),
    )

    if not front_detections:
        raise HTTPException(
            status_code=400,
            detail="No front-side citizenship regions were detected in the uploaded image.",
        )

    if not back_detections:
        raise HTTPException(
            status_code=400,
            detail="No back-side citizenship fields were detected in the uploaded image.",
        )

    parsed_fields = parse_back(
        image_path=str(back_image_path),
        detector=detector,
        output_dir=str(back_crop_dir),
    )

    accuracy_report = _compute_field_accuracy(
        full_name=full_name,
        date_of_birth=date_of_birth,
        citizenship_number=citizenship_number,
        permanent_address=permanent_address,
        parsed_fields=parsed_fields,
    )

    print("[OCR-ACCURACY] Field Accuracy:", accuracy_report["field_accuracy"])
    print("[OCR-ACCURACY] Overall Accuracy:", accuracy_report["overall_accuracy"])

    doc_face_crop_path = _pick_photo_crop(front_detections)
    doc_logo_crop_path = _pick_logo_crop(front_detections)

    face_similarity: dict[str, Any] | None = None
    if doc_face_crop_path:
        try:
            face_similarity = _run_face_similarity(doc_face_crop_path, str(selfie_path))
        except ValueError as exc:
            face_similarity = {
                "error": str(exc),
                "doc_face_crop_path": doc_face_crop_path,
                "selfie_path": str(selfie_path),
            }
        except Exception as exc:
            # Do not fail document OCR flow because of optional face matcher issues.
            face_similarity = {"error": f"Face similarity failed: {exc}"}
    else:
        face_similarity = {"error": "No citizenship photo crop found in front detections."}

    stamp_similarity: dict[str, Any] | None = None
    if doc_logo_crop_path:
        try:
            stamp_similarity = _compute_stamp_similarity(doc_logo_crop_path)
        except ValueError as exc:
            stamp_similarity = {"error": str(exc)}
        except Exception as exc:
            stamp_similarity = {"error": f"Stamp similarity failed: {exc}"}
    else:
        stamp_similarity = {"error": "No government logo/stamp crop found in front detections."}

    try:
        passive_liveness = _run_passive_liveness(str(selfie_path))
    except Exception as exc:
        passive_liveness = {"error": f"Passive liveness failed: {exc}"}

    try:
        document_tampering = _compute_document_tampering(
            front_image_path=str(front_image_path),
            back_image_path=str(back_image_path),
        )
    except Exception as exc:
        document_tampering = {"error": f"Document tampering failed: {exc}"}

    # ✅ SAFE TERMINAL LOGGING (won't crash)
    if face_similarity is None:
        print("[FACE-SIMILARITY] Skipped: embed/match modules not available.")
    elif isinstance(face_similarity, dict) and "error" in face_similarity:
        print(f"[FACE-SIMILARITY] Failed: {face_similarity['error']}")
    else:
        print(
            "[FACE-SIMILARITY] "
            f"doc={doc_face_crop_path} "
            f"selfie={selfie_path} "
            f"score={face_similarity['similarity']:.6f} "
            f"match={face_similarity['match']} "
            f"threshold={face_similarity['threshold_used']}"
        )

    if stamp_similarity is None:
        print("[STAMP-SIMILARITY] Skipped.")
    elif isinstance(stamp_similarity, dict) and "error" in stamp_similarity:
        print(f"[STAMP-SIMILARITY] Failed: {stamp_similarity['error']}")
    else:
        print(
            "[STAMP-SIMILARITY] "
            f"detected={doc_logo_crop_path} "
            f"matched={stamp_similarity['best_match']['standard_stamp_path']} "
            f"score={stamp_similarity['best_match']['score']:.4f} "
            f"same={stamp_similarity['is_same_stamp']} "
            f"threshold={stamp_similarity['threshold_used']}"
        )

    if isinstance(passive_liveness, dict) and "error" in passive_liveness:
        print(f"[PASSIVE-LIVENESS] Failed: {passive_liveness['error']}")
        _print_terminal_result("PASSIVE-LIVENESS", passive_liveness)
    else:
        print(
            "[PASSIVE-LIVENESS] "
            f"selfie={selfie_path} "
            f"live_probability={passive_liveness['live_probability']:.4f} "
            f"spoof_probability={passive_liveness['spoof_probability']:.4f} "
            f"passed={passive_liveness['passed']} "
            f"threshold={passive_liveness['threshold_used']}"
        )
        _print_terminal_result("PASSIVE-LIVENESS", passive_liveness)

    if isinstance(document_tampering, dict) and "error" in document_tampering:
        print(f"[DOCUMENT-TAMPERING] Failed: {document_tampering['error']}")
        _print_terminal_result("DOCUMENT-TAMPERING", document_tampering)
    else:
        print(
            "[DOCUMENT-TAMPERING] "
            f"front_tampered={document_tampering['front']['is_tampered']} "
            f"back_tampered={document_tampering['back']['is_tampered']} "
            f"overall_genuine_score={document_tampering['overall_genuine_score']:.4f}"
        )
        _print_terminal_result("DOCUMENT-TAMPERING", document_tampering)

    kyc_decision = _compute_kyc_decision(
        accuracy_report=accuracy_report,
        face_similarity=face_similarity,
        stamp_similarity=stamp_similarity,
        passive_liveness=passive_liveness,
        document_tampering=document_tampering,
    )

    return {
        "status": kyc_decision["status"],
        "approved": kyc_decision["approved"],
        "final_score": kyc_decision["overall_score"],
        "overall_threshold": kyc_decision["threshold"],
        "weights": kyc_decision["weights"],
        "component_scores": kyc_decision["component_scores"],
        "decision_reason": kyc_decision["decision_reason"],
        "message": "Citizenship front/back processed and OCR extracted.",
        "uploaded_front_image": str(front_image_path),
        "uploaded_back_image": str(back_image_path),
        "uploaded_selfie_image": str(selfie_path),
        "front_crop_directory": str(front_crop_dir),
        "back_crop_directory": str(back_crop_dir),
        "front_detections": front_detections,
        "back_detections": back_detections,
        "parsed_fields": parsed_fields,
        "accuracy_report": accuracy_report,
        "doc_face_crop": doc_face_crop_path,
        "doc_logo_crop": doc_logo_crop_path,
        "face_similarity": face_similarity,
        "stamp_similarity": stamp_similarity,
        "passive_liveness": passive_liveness,
        "document_tampering": document_tampering,
    }


@app.post("/api/liveness/run")
def run_liveness():
    if not LIVENESS_SCRIPT.exists():
        raise HTTPException(status_code=500, detail="ml/liveness-service/liveness.py not found")

    cmd = [
        sys.executable,
        str(LIVENESS_SCRIPT),
        "--json",
        "--output-dir",
        str(OUTPUT_DIR),
    ]

    completed = subprocess.run(
        cmd,
        cwd=str(ROOT_DIR),
        capture_output=True,
        text=True,
        check=False,
    )

    stdout = completed.stdout.strip().splitlines()
    result_line = stdout[-1] if stdout else "{}"

    try:
        result = json.loads(result_line)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Invalid liveness response: {result_line}",
        ) from exc

    image_path = result.get("image_path")
    image_data_url = None

    if image_path and os.path.exists(image_path):
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode("utf-8")
        image_data_url = f"data:image/jpeg;base64,{encoded}"

    return {
        "passed": bool(result.get("passed")),
        "message": result.get("message", "Liveness failed"),
        "image": image_data_url,
        "return_code": completed.returncode,
        "stderr": completed.stderr.strip(),
    }
