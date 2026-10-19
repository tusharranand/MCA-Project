from __future__ import annotations

from typing import Any, Dict


def audit_dataset(df) -> Dict[str, Any]:
    """Summarize the churn dataset for data-quality and model pipeline review."""
    target = "Churned"
    missing_values = df.isna().sum().to_dict()
    duplicate_rows = int(df.duplicated().sum())
    target_series = df[target].astype(str)
    target_distribution = target_series.value_counts(dropna=False).to_dict()
    class_balance = {
        "positive": int(target_series.eq("Yes").sum()),
        "negative": int(target_series.eq("No").sum()),
        "positive_rate": round(float(target_series.eq("Yes").mean()), 4),
    }

    return {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "duplicate_rows": duplicate_rows,
        "missing_values": missing_values,
        "target_distribution": target_distribution,
        "class_balance": class_balance,
    }
