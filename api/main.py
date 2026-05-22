"""FastAPI application entrypoint (minimal)."""
from fastapi import FastAPI, HTTPException
from .schemas import PredictRequest, PredictResponse
from .predictor import predictor

app = FastAPI(title="Diabetes Readmission Predictor")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        pred, prob = predictor.predict(req.features)
        return PredictResponse(prediction=pred, probability=prob)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
