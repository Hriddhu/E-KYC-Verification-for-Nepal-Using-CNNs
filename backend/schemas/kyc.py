from datetime import date
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict
import re

class GenderEnum(str, Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"

# frontend form request
class KYCFormData(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    # Using Field constraints to eliminate basic length validation code
    full_name: str = Field(..., min_length=2, max_length=100)
    date_of_birth: date  
    gender: GenderEnum  
    citizenship_number: str
    permanent_address: str = Field(..., min_length=5, max_length=300)
    current_address: str = Field(..., min_length=5, max_length=300)

    @field_validator("full_name")
    def validate_name_characters(cls, v: str) -> str:
        if not re.match(r"^[a-zA-Z\s]+$", v):
            raise ValueError("Name must contain only letters and spaces")
        return v

    @field_validator("citizenship_number")
    def validate_citizenship(cls, v: str) -> str:
        # Assuming format: XX-XX-XX-XXXXX (For newer citizenships)
        if not re.match(r"^\d{2}-\d{2}-\d{2}-\d{5}$", v):
            raise ValueError("Invalid citizenship number format. Expected XX-XX-XX-XXXXX")
        return v

    @field_validator("date_of_birth")
    def validate_age_limit(cls, v: date) -> date:
        today = date.today()
        age = today.year - v.year - ((today.month, today.day) < (v.month, v.day))
        if age < 16:
            raise ValueError("Applicant must be at least 16 years old")
        if age > 120:
            raise ValueError("Invalid date of birth")
        return v


#  Component level scores 
class ComponentScores(BaseModel):
    # Enforce scores to be mathematically bound between 0.0 and 1.0 (or 0 and 100)
    face_similarity: float = Field(..., ge=0.0, le=1.0)
    stamp_similarity: float = Field(..., ge=0.0, le=1.0)
    ocr_accuracy: float = Field(..., ge=0.0, le=1.0)
    passive_liveness: float = Field(..., ge=0.0, le=1.0)
    document_tampering: float = Field(..., ge=0.0, le=1.0)


#  OCR extracted values 
class ExtractedFields(BaseModel):
    full_name: Optional[str] = None
    date_of_birth: Optional[date] = None  # Unified type matching KYCFormData
    citizenship_number: Optional[str] = None
    permanent_address: Optional[str] = None


#  Final KYC response sent to frontend 
class KYCResponse(BaseModel):
    status: str  # Considering an Enum here as well (e.g., "completed", "failed", "flagged")
    approved: bool
    final_score: float = Field(..., ge=0.0, le=1.0)
    decision_reason: str
    component_scores: ComponentScores
    extracted_fields: Optional[ExtractedFields] = None