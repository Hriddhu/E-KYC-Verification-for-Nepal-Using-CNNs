from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

# ML service paths
ROOT_DIR = Path(__file__).resolve().parents[1]
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

OCR_SERVICE_SRC_DIR = ROOT_DIR / "ml" / "ocr-service"
if str(OCR_SERVICE_SRC_DIR) not in sys.path:
    sys.path.append(str(OCR_SERVICE_SRC_DIR))

FACE_SERVICE_SRC_DIR = ROOT_DIR / "ml" / "face-service"

if str(FACE_SERVICE_SRC_DIR) not in sys.path:
    sys.path.append(str(FACE_SERVICE_SRC_DIR))

# Model file paths


# Thresholds


# KYC weights