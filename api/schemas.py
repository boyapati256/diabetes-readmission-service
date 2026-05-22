"""Pydantic schemas for API requests/responses."""
from pydantic import BaseModel
from typing import Dict, Any


class PredictRequest(BaseModel):
    features: Dict[str, Any]


class PredictResponse(BaseModel):
    prediction: int
    probability: float
