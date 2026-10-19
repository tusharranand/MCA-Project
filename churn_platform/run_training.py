import json
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.data_loader import load_customer_dataset
from app.data_quality import audit_dataset
from app.modeling import build_model_pipeline, prepare_feature_matrix, train_candidate_models


def main() -> None:
    dataset_path = PROJECT_ROOT / "Client Data Set for Student Project Workbook.xlsx"
    df = load_customer_dataset(str(dataset_path))
    quality = audit_dataset(df)
    train_summary = train_candidate_models(df)
    X, y = prepare_feature_matrix(df)
    selected_pipeline = build_model_pipeline(X)
    selected_pipeline.fit(X, y)

    artifact_dir = ROOT / "artifacts"
    artifact_dir.mkdir(exist_ok=True)
    artifact_path = artifact_dir / "churn_model.joblib"
    joblib.dump(
        {
            "pipeline": selected_pipeline,
            "feature_columns": list(X.columns),
            "model_version": "v1.0.0",
            "risk_threshold": 0.4,
            "high_risk_threshold": 0.7,
        },
        artifact_path,
    )

    report = {
        "dataset_path": str(dataset_path),
        "data_quality": quality,
        "model_comparison": train_summary["model_comparison"],
        "selected_model": "logistic_regression",
        "selection_note": "Provisional academic default based on the highest holdout F1 in this initial comparison; business metric and threshold are not confirmed.",
        "model_artifact": str(artifact_path),
        "feature_shape": {
            "train_shape": list(train_summary["train_shape"]),
            "test_shape": list(train_summary["test_shape"]),
        },
    }

    output_dir = ROOT / "reports"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "model_report.json"
    output_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved model report to {output_file}")
    print(f"Saved model artifact to {artifact_path}")


if __name__ == "__main__":
    main()
