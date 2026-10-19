import os

from churn_platform.app.data_loader import load_customer_dataset
from churn_platform.app.data_quality import audit_dataset


def test_load_customer_dataset_reads_real_workbook():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    workbook = os.path.join(root, "Client Data Set for Student Project Workbook.xlsx")
    df = load_customer_dataset(workbook)
    assert df.shape[0] == 2200
    assert df.shape[1] == 18
    assert "Churned" in df.columns
    assert "Customer_ID" in df.columns


def test_audit_dataset_reports_target_distribution_and_quality():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    workbook = os.path.join(root, "Client Data Set for Student Project Workbook.xlsx")
    df = load_customer_dataset(workbook)
    summary = audit_dataset(df)
    assert summary["n_rows"] == 2200
    assert summary["n_columns"] == 18
    assert "target_distribution" in summary
    assert "missing_values" in summary
    assert "class_balance" in summary
