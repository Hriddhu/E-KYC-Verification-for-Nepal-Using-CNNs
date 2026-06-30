from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

# ML service paths
UPLOAD_DIR = ROOT_DIR / "ml" / "docservice" / "uploads"
CROP_OUTPUT_DIR = ROOT_DIR / "ml" / "docservice" / "crops"
STANDARD_STAMP_DIR = ROOT_DIR / "ml" / "docservice" / "standard_stamps"
LIVENESS_OUTPUT_DIR = ROOT_DIR / "ml" / "liveness_service" / "extracted_faces"


# Model file paths

TAMPERING_MODEL_PATH = ROOT_DIR / "ml" / "tampering" / "best_model.pth"
PASSIVE_LIVENESS_MODEL_PATH = ROOT_DIR / "ml" / "liveness_service" / "final_model.pt"
YOLO_FRONT_MODEL_PATH = ROOT_DIR / "ml" / "docservice" / "weights" /"frontbest.pt"
YOLO_BACK_MODEL_PATH = ROOT_DIR / "ml" / "docservice" / "weights" / "backbest.pt"
FACE_MODEL_NAME = "buffalo_l"
EASYOCR_LANGUAGES = ["en"]     

# Thresholds
KYC_APPROVAL_THRESHOLD = 0.55
PASSIVE_LIVENESS_THRESHOLD = 0.5
TAMPERING_THRESHOLD = 0.78
FACE_MATCH_THRESHOLD = 0.60   
STAMP_SIMILARITY_THRESHOLD = 0.72  


# KYC weights
KYC_WEIGHTS = {
    "face_similarity": 0.3,
    "stamp_similarity": 0.10,
    "ocr_accuracy": 0.20,
    "passive_liveness": 0.2,
    "document_tampering": 0.2,
}