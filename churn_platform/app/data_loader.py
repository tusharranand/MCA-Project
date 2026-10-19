from pathlib import Path
from typing import Optional

import pandas as pd


def load_customer_dataset(path: Optional[str] = None) -> pd.DataFrame:
    """Load the approved customer dataset and perform a lightweight schema check."""
    if path is None:
        default_candidates = [
            Path(__file__).resolve().parents[2] / "Client Data Set for Student Project Workbook.xlsx",
            Path(__file__).resolve().parents[2] / "data" / "customer_data.csv",
        ]
        existing = [candidate for candidate in default_candidates if candidate.exists()]
        if not existing:
            raise FileNotFoundError("No approved customer dataset was found in the project root.")
        chosen = existing[0]
    else:
        chosen = Path(path)

    if not chosen.exists():
        raise FileNotFoundError(f"Dataset not found at {chosen}")

    if chosen.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(chosen, sheet_name="Customer Intelligence Data")
    else:
        df = pd.read_csv(chosen)

    required_columns = {"Customer_ID", "Churned"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    return df
