from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    app_name: str = "AI-Powered Customer Churn Prediction Platform"
    python_version: str = "3.11"
    model_version: str = "v1.0.0"
    default_dataset_path: str = r"C:\Users\tushar.anand\Downloads\MCA Project\Client Data Set for Student Project Workbook.xlsx"
    target_column: str = "Churned"
    identifier_column: str = "Customer_ID"
    risk_score_column: str = "Previous_Churn_Risk_Score"
