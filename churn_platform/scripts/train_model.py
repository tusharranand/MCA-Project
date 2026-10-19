from __future__ import annotations

import json
from pathlib import Path

from app.data_loader import load_customer_dataset
from app.data_quality import audit_dataset
from app.modeling import train_candidate_models


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    workspace_root = project_root.parent
    dataset_path = workspace_root / "Client Data Set for Student Project Workbook.xlsx"

    df = load_customer_dataset(str(dataset_path))
    quality = audit_dataset(df)
    model_summary = train_candidate_models(df)

    report = {
        "dataset_path": str(dataset_path),
        "data_quality": quality,
        "model_comparison": model_summary["model_comparison"],
        "feature_shape": {
            "train_shape": model_summary["train_shape"],
            "test_shape": model_summary["test_shape"],
        },
    }

    output_dir = project_root / "reports"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "model_report.json"
    output_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved model summary to {output_file}")


if __name__ == "__main__":
    main()
