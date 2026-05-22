import yaml
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from api.schemas import PredictionRequest, PredictionResponse, HealthResponse
from api.predictor import ReadmissionPredictor
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Global predictor instance ─────────────────────────────────────────────────
# Loaded once at startup, reused for every request
predictor: ReadmissionPredictor = None


def load_config(config_path: str = "configs/train.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: load model artifacts into memory once.
    Shutdown: clean up if needed.
    Using lifespan instead of deprecated @app.on_event.
    """
    global predictor

    logger.info("Starting up — loading model artifacts...")
    config = load_config()
    artifact_dir = config["api"].get(
        "model_artifact_path", "models/v1/"
    )

    try:
        predictor = ReadmissionPredictor(artifact_dir=artifact_dir)
        logger.info("Model loaded successfully. API is ready.")
    except FileNotFoundError as e:
        logger.error(f"Failed to load model: {e}")
        # Don't crash the app — health endpoint will report not ready

    yield  # app runs here

    logger.info("Shutting down API.")


# ── App definition ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="Diabetes Readmission Prediction API",
    description=(
        "Predicts the probability of a diabetic patient being readmitted "
        "to hospital within 30 days of discharge. "
        "Built on the UCI Diabetes 130-US Hospitals dataset."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Middleware: request timing ─────────────────────────────────────────────────
@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    """Adds X-Response-Time header to every response for latency monitoring."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    response.headers["X-Response-Time-Ms"] = str(elapsed_ms)
    return response


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Ops"])
def health_check():
    """
    Liveness + readiness check.
    Returns model_loaded=false if artifacts failed to load at startup.
    """
    loaded = predictor is not None and predictor.is_loaded
    version = predictor.model_version if loaded else "unavailable"

    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        model_version=version,
        message="Ready" if loaded else "Model not loaded — run training first.",
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(request: PredictionRequest):
    """
    Predict 30-day readmission probability for a single patient encounter.

    - **prediction**: 1 = readmitted within 30 days, 0 = not readmitted
    - **probability**: model confidence (0.0 – 1.0)
    - **risk_level**: Low / Medium / High based on probability bands
    - **model_version**: artifact version used for this prediction
    """
    if predictor is None or not predictor.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Run training first: python scripts/train.py",
        )

    try:
        result = predictor.predict(request.model_dump())
        return PredictionResponse(**result)

    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


@app.get("/", tags=["Ops"])
def root():
    return {
        "service": "diabetes-readmission-service",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }
