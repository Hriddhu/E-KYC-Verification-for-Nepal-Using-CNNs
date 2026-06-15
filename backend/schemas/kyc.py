from pydantic import BaseModel, field_validator
from typing import Optional
import re


class KYCFormData(BaseModel):
    full_name: str
    date_of_birth: str
    gender: str
    citizenship_number: str
    permanent_address: str
    current_address: str

    @field_validator("full_name")
    def validate_name(cls, v):
        v = v.strip()
        if len(v) < 2 or len(v) > 100:
            raise ValueError("Name must be between 2 and 100 characters")
        if not re.match(r"^[a-zA-Z\s]+$", v):
            raise ValueError("Name must contain only letters and spaces")
        return v

    @field_validator("citizenship_number")
    def validate_citizenship(cls, v):
        v = v.strip()
        if not re.match(r"^\d{2}-\d{2}-\d{2}-\d{5}$", v):
            raise ValueError("Invalid citizenship number format. Expected XX-XX-XX-XXXXX")
        return v

    @field_validator("date_of_birth")
    def validate_dob(cls, v):
        v = v.strip()
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", v):
            raise ValueError("Date of birth must be in YYYY-MM-DD format")
        return v

    @field_validator("gender")
    def validate_gender(cls, v):
        v = v.strip().lower()
        if v not in {"male", "female", "other"}:
            raise ValueError("Gender must be male, female, or other")
        return v

    @field_validator("permanent_address", "current_address")
    def validate_address(cls, v):
        v = v.strip()
        if len(v) < 5 or len(v) > 300:
            raise ValueError("Address must be between 5 and 300 characters")
        return v


# ── Component level scores ───────────────────────────────
class ComponentScores(BaseModel):
    face_similarity: float
    stamp_similarity: float
    ocr_accuracy: float
    passive_liveness: float
    document_tampering: float


# ── OCR extracted values ─────────────────────────────────
class ExtractedFields(BaseModel):
    full_name: Optional[str] = None
    date_of_birth: Optional[str] = None
    citizenship_number: Optional[str] = None
    permanent_address: Optional[str] = None


# ── Final KYC response sent to frontend ─────────────────
class KYCResponse(BaseModel):
    status: str
    approved: bool
    final_score: float
    decision_reason: str
    component_scores: ComponentScores
    extracted_fields: Optional[ExtractedFields] = None