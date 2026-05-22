from pydantic import BaseModel, Field, field_validator
from typing import Optional
from enum import Enum


# ── Enums for constrained categorical fields ─────────────────────────────────

class RaceEnum(str, Enum):
    caucasian = "Caucasian"
    african_american = "AfricanAmerican"
    hispanic = "Hispanic"
    asian = "Asian"
    other = "Other"
    missing = "Missing"


class GenderEnum(str, Enum):
    male = "Male"
    female = "Female"
    unknown = "Unknown/Invalid"


class InsulinEnum(str, Enum):
    no = "No"
    steady = "Steady"
    up = "Up"
    down = "Down"
    missing = "Missing"


class A1CResultEnum(str, Enum):
    norm = "Norm"
    greater7 = ">7"
    greater8 = ">8"
    none = "None"
    missing = "Missing"


class MaxGluSerumEnum(str, Enum):
    norm = "Norm"
    greater200 = ">200"
    greater300 = ">300"
    none = "None"
    missing = "Missing"


class MedChangeEnum(str, Enum):
    ch = "Ch"
    no = "No"
    missing = "Missing"


class DiabetesMedEnum(str, Enum):
    yes = "Yes"
    no = "No"
    missing = "Missing"


# ── Request Schema ────────────────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    """
    Single patient encounter record for readmission prediction.
    All fields mirror the UCI dataset after preprocessing.
    Optional fields default to 'Missing' — the model handles them gracefully.
    """

    # Numeric fields
    time_in_hospital: int = Field(..., ge=1, le=14,
                                  description="Days in hospital (1–14)")
    num_lab_procedures: int = Field(..., ge=0, le=132,
                                    description="Number of lab procedures")
    num_procedures: int = Field(..., ge=0, le=6,
                                description="Number of non-lab procedures")
    num_medications: int = Field(..., ge=0, le=81,
                                 description="Number of distinct medications")
    number_outpatient: int = Field(..., ge=0,
                                   description="Outpatient visits in prior year")
    number_emergency: int = Field(..., ge=0,
                                  description="Emergency visits in prior year")
    number_inpatient: int = Field(..., ge=0,
                                  description="Inpatient visits in prior year")
    number_diagnoses: int = Field(..., ge=1, le=16,
                                  description="Number of diagnoses entered")
    age: int = Field(..., ge=0, le=100,
                     description="Patient age (numeric midpoint)")

    # Categorical fields
    race: RaceEnum = Field(RaceEnum.missing,        description="Patient race")
    gender: GenderEnum = Field(GenderEnum.unknown,
                               description="Patient gender")
    insulin: InsulinEnum = Field(
        InsulinEnum.missing,     description="Insulin dosage change")
    change: MedChangeEnum = Field(
        MedChangeEnum.missing,   description="Change in diabetic medications")
    diabetesMed: DiabetesMedEnum = Field(
        DiabetesMedEnum.missing, description="Diabetic medication prescribed")
    A1Cresult: A1CResultEnum = Field(
        A1CResultEnum.missing,   description="A1C test result")
    max_glu_serum: MaxGluSerumEnum = Field(
        MaxGluSerumEnum.missing,  description="Max glucose serum result")

    # Diagnosis categories (output of ICD-9 grouping in preprocessing)
    diag_1: Optional[str] = Field(
        "Other", description="Primary diagnosis category")
    diag_2: Optional[str] = Field(
        "Other", description="Secondary diagnosis category")
    diag_3: Optional[str] = Field(
        "Other", description="Additional diagnosis category")

    # Admission context
    admission_type_id: Optional[int] = Field(1, ge=1, le=8)
    discharge_disposition_id: Optional[int] = Field(1, ge=1, le=30)
    admission_source_id: Optional[int] = Field(7, ge=1, le=26)

    # Medication fields (common ones — all default to No)
    metformin: Optional[str] = Field("No")
    glipizide: Optional[str] = Field("No")
    glyburide: Optional[str] = Field("No")
    pioglitazone: Optional[str] = Field("No")
    rosiglitazone: Optional[str] = Field("No")
    acarbose: Optional[str] = Field("No")

    @field_validator("diag_1", "diag_2", "diag_3")
    @classmethod
    def validate_diagnosis_category(cls, v):
        valid = {
            "Circulatory", "Respiratory", "Digestive", "Diabetes",
            "Injury", "Musculoskeletal", "Genitourinary", "Neoplasms",
            "External_or_Supplemental", "Other", "Unknown", "Missing"
        }
        if v not in valid:
            return "Other"   # graceful fallback for unknown codes
        return v

    model_config = {"use_enum_values": True}


# ── Response Schema ───────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    """
    Prediction result for a single patient encounter.
    """
    prediction: int = Field(
        ..., description="Binary prediction: 1 = readmitted within 30 days, 0 = not")
    probability: float = Field(
        ..., description="Model confidence score (probability of readmission)")
    risk_level: str = Field(...,
                            description="Human-readable risk band: Low / Medium / High")
    model_version: str = Field(...,
                               description="Model artifact version used for this prediction")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    message: str
