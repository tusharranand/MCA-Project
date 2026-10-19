from churn_platform.app.services.prediction_service import PredictionService
from churn_platform.app.modeling import build_model_pipeline


def test_prediction_service_classifies_risk_bands():
    service = PredictionService(threshold=0.4)
    assert service.classify_risk(0.15) == "Low"
    assert service.classify_risk(0.45) == "Medium"
    assert service.classify_risk(0.8) == "High"


def test_prediction_service_scores_outputs_consistent_schema():
    service = PredictionService(threshold=0.4)
    output = service.score({"Age": 40, "Tenure_Months": 18, "Monthly_Spend_INR": 300})
    assert set(output.keys()) >= {"risk_probability", "risk_band", "model_version", "explanation"}
    assert 0.0 <= output["risk_probability"] <= 1.0
    assert output["risk_band"] in {"Low", "Medium", "High"}


def test_prediction_service_uses_trained_pipeline_with_missing_optional_fields():
    import pandas as pd

    training_features = pd.DataFrame(
        {
            "Age": [24, 30, 42, 55, 36, 63],
            "Region": ["North", "East", "West", "Central", "East", "North"],
        }
    )
    labels = pd.Series(["No", "No", "Yes", "Yes", "No", "Yes"])
    pipeline = build_model_pipeline(training_features)
    pipeline.fit(training_features, labels)
    service = PredictionService(model=pipeline, feature_columns=list(training_features.columns))

    result = service.score({"Age": 44})

    assert 0.0 <= result["risk_probability"] <= 1.0
    assert result["risk_band"] in {"Low", "Medium", "High"}
    assert "saved churn model pipeline" in result["explanation"][0]
