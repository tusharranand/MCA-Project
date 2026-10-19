from typing import Dict, List

from fastapi import FastAPI, HTTPException

from app.config import AppConfig
from app.schemas import PredictionRequest, PredictionResponse
from app.services.prediction_service import PredictionService

app = FastAPI(title="Churn Prediction API", version=AppConfig().model_version)
service = PredictionService(model_version=AppConfig().model_version)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "app": AppConfig().app_name, "model_version": AppConfig().model_version}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    if not payload.features:
        raise HTTPException(status_code=400, detail="Features cannot be empty.")

    output = service.score(payload.features)
    return PredictionResponse(
        customer_id=payload.customer_id,
        risk_probability=output["risk_probability"],
        risk_band=output["risk_band"],
        model_version=output["model_version"],
        explanation=output["explanation"],
        validation_errors=[],
    )


@app.get("/customers")
def list_customers() -> List[Dict[str, str]]:
    return [{"customer_id": "demo_customer_001", "status": "review-required"}]


@app.get("/customers/{customer_id}")
def get_customer(customer_id: str) -> Dict[str, str]:
    return {"customer_id": customer_id, "status": "review-required", "risk_band": "Medium"}
