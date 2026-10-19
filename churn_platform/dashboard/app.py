from __future__ import annotations

from html import escape
import sys
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.data_loader import load_customer_dataset
from app.modeling import prepare_feature_matrix
from app.services.prediction_service import PredictionService

DATASET_PATH = PROJECT_ROOT.parent / "Client Data Set for Student Project Workbook.xlsx"
MODEL_PATH = PROJECT_ROOT / "artifacts" / "churn_model.joblib"

st.set_page_config(page_title="Churn Risk Review", layout="wide")
st.title("AI-Powered Customer Churn Prediction Platform")
st.caption("Customer risk review | Advisory estimates for analyst review")


def render_pinned_prediction_table(history: pd.DataFrame, columns: list[str]) -> None:
        table_columns = ["Customer_ID", *columns, "Model version", "Risk percent", "Risk band"]
        table_min_width = 150 + (len(columns) * 150) + 130 + 130 + 105
        rows = history.reindex(columns=table_columns).fillna("").to_dict(orient="records")
        headers = "".join(f"<th>{escape(column)}</th>" for column in table_columns)
        body_rows = []
        for record in rows:
                cells = "".join(f"<td>{escape(str(record[column]))}</td>" for column in table_columns)
                body_rows.append(f"<tr>{cells}</tr>")

        table_html = f"""
        <style>
            .prediction-table-scroll {{ --table-cell-bg: #0e1117; --table-header-bg: #262730; --table-alt-bg: #171b24; --table-border: #31333f; max-height: 480px; overflow: auto; border: 1px solid var(--table-border); color: inherit; background: var(--table-cell-bg); }}
            .prediction-table {{ border-collapse: separate; border-spacing: 0; table-layout: fixed; width: {table_min_width}px; min-width: {table_min_width}px; font-size: 0.875rem; color: inherit; }}
            .prediction-table th, .prediction-table td {{ box-sizing: border-box; width: 150px; min-width: 150px; max-width: 150px; overflow: hidden; padding: 0.55rem 0.75rem; white-space: nowrap; text-overflow: ellipsis; text-align: left; border-bottom: 1px solid var(--table-border); background: var(--table-cell-bg); color: inherit; }}
            .prediction-table thead th {{ position: sticky; top: 0; z-index: 4; background: var(--table-header-bg); font-weight: 600; }}
            .prediction-table th:first-child, .prediction-table td:first-child {{ position: sticky; left: 0; z-index: 3; width: 150px; min-width: 150px; background: var(--table-cell-bg); }}
            .prediction-table thead th:first-child {{ z-index: 6; background: var(--table-header-bg); }}
            .prediction-table th:nth-last-child(3), .prediction-table td:nth-last-child(3) {{ width: 130px; min-width: 130px; max-width: 130px; }}
            .prediction-table th:nth-last-child(2), .prediction-table td:nth-last-child(2) {{ position: sticky; right: 105px; z-index: 3; width: 130px; min-width: 130px; max-width: 130px; background: var(--table-cell-bg); }}
            .prediction-table th:last-child, .prediction-table td:last-child {{ position: sticky; right: 0; z-index: 3; width: 105px; min-width: 105px; max-width: 105px; background: var(--table-cell-bg); }}
            .prediction-table thead th:nth-last-child(2), .prediction-table thead th:last-child {{ z-index: 6; background: var(--table-header-bg); }}
            .prediction-table tbody tr:nth-child(even) td {{ background: var(--table-alt-bg); }}
            .prediction-table tbody tr:nth-child(even) td:first-child,
            .prediction-table tbody tr:nth-child(even) td:nth-last-child(2),
            .prediction-table tbody tr:nth-child(even) td:last-child {{ background: var(--table-alt-bg); }}
        </style>
        <div class="prediction-table-scroll">
            <table class="prediction-table">
                <thead><tr>{headers}</tr></thead>
                <tbody>{''.join(body_rows)}</tbody>
            </table>
        </div>
        """
        st.markdown(table_html, unsafe_allow_html=True)


@st.cache_resource
def load_resources() -> tuple[pd.DataFrame, PredictionService, list[str]]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model artifact not found. Run `python run_training.py` from the churn_platform folder first.")

    artifact: Dict[str, Any] = joblib.load(MODEL_PATH)
    dataset = load_customer_dataset(str(DATASET_PATH))
    feature_columns, _ = prepare_feature_matrix(dataset)
    service = PredictionService(
        model_version=artifact["model_version"],
        threshold=artifact["risk_threshold"],
        high_risk_threshold=artifact["high_risk_threshold"],
        model=artifact["pipeline"],
        feature_columns=artifact["feature_columns"],
    )
    return dataset, service, list(feature_columns.columns)


try:
    source_data, prediction_service, feature_columns = load_resources()
except (FileNotFoundError, KeyError, ValueError) as error:
    st.error(str(error))
    st.stop()

st.info("Risk bands use provisional 40% and 70% cutoffs. Estimates are not calibrated or validated for customer interventions.")

selected_fields = st.multiselect(
    "Optional customer fields",
    options=feature_columns,
    help="Choose any fields available for this customer. Unselected fields are imputed by the saved model pipeline.",
)

with st.form("customer_review"):
    st.subheader("Customer details")
    customer_id = st.text_input("Customer ID", value="demo_customer_001")
    provided_features: Dict[str, Any] = {}
    columns = st.columns(2)

    for index, feature in enumerate(selected_fields):
        source_column = source_data[feature]
        with columns[index % 2]:
            if pd.api.types.is_numeric_dtype(source_column):
                minimum = float(source_column.min())
                maximum = float(source_column.max())
                default = float(source_column.median())
                if pd.api.types.is_integer_dtype(source_column):
                    provided_features[feature] = st.number_input(
                        feature,
                        min_value=int(minimum),
                        max_value=int(maximum),
                        value=int(round(default)),
                        step=1,
                        key=f"value-{feature}",
                    )
                else:
                    provided_features[feature] = st.number_input(
                        feature,
                        min_value=minimum,
                        max_value=maximum,
                        value=default,
                        step=0.1,
                        key=f"value-{feature}",
                    )
            else:
                choices = sorted(source_column.dropna().astype(str).unique().tolist())
                if choices:
                    provided_features[feature] = st.selectbox(
                        feature,
                        choices,
                        key=f"value-{feature}",
                    )

    submitted = st.form_submit_button("Score customer", type="primary")

if submitted:
    result = prediction_service.score(provided_features)
    customer_result = {
        "Customer_ID": customer_id.strip() or f"customer_{len(st.session_state.get('predictions', [])) + 1:04d}",
        **{feature: provided_features.get(feature, "Not provided") for feature in feature_columns},
        "Model version": result["model_version"],
        "Risk percent": f"{result['risk_probability']:.1%}",
        "Risk band": result["risk_band"],
    }
    st.session_state.setdefault("predictions", []).append(customer_result)
    st.success(f"Estimated churn risk: {result['risk_probability']:.1%} ({result['risk_band']})")
    st.caption("Model estimates are advisory; review the customer record before deciding on any action.")

predictions = st.session_state.get("predictions", [])
if predictions:
    history = pd.DataFrame(predictions)
    st.subheader("Predictions in this session")

    chart_columns, counts_column = st.columns([2, 1])
    with chart_columns:
        st.vega_lite_chart(
            history["Risk band"].value_counts().reindex(["Low", "Medium", "High"], fill_value=0).rename_axis("Risk band").reset_index(name="Customers"),
            {
                "mark": {"type": "arc", "outerRadius": 120},
                "encoding": {
                    "theta": {"field": "Customers", "type": "quantitative"},
                    "color": {
                        "field": "Risk band",
                        "type": "nominal",
                        "scale": {"domain": ["Low", "Medium", "High"], "range": ["#2E7D5B", "#D29A29", "#BD4C42"]},
                    },
                    "tooltip": [
                        {"field": "Risk band", "type": "nominal"},
                        {"field": "Customers", "type": "quantitative"},
                    ],
                },
                "height": 300,
            },
            use_container_width=True,
        )
    with counts_column:
        band_counts = history["Risk band"].value_counts().reindex(["Low", "Medium", "High"], fill_value=0)
        st.metric("Customers scored", len(history))
        for band, count in band_counts.items():
            st.metric(f"{band} risk", int(count))

    st.subheader("Predicted customer information")
    render_pinned_prediction_table(history, feature_columns)
    if st.button("Clear session predictions"):
        st.session_state.predictions = []
        st.rerun()
else:
    st.caption("Submit a customer record to populate the risk-band chart and prediction table.")
