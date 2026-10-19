from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from app.config import AppConfig
from app.data_loader import load_customer_dataset


def prepare_feature_matrix(df: pd.DataFrame, exclude_columns: List[str] | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Build a leakage-safe feature matrix and target series for churn modeling."""
    if exclude_columns is None:
        exclude_columns = [AppConfig().identifier_column, AppConfig().risk_score_column]

    data = df.copy()
    target = AppConfig().target_column
    X = data.drop(columns=[col for col in exclude_columns if col in data.columns] + [target], errors="ignore")
    y = data[target].astype(str)
    return X, y


def build_model_pipeline(X: pd.DataFrame) -> Pipeline:
    categorical_features = list(X.select_dtypes(include=["object", "category", "bool"]).columns)
    numeric_features = list(X.select_dtypes(exclude=["object", "category", "bool"]).columns)

    transformers = []
    if numeric_features:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        )
    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            )
        )

    return Pipeline(
        steps=[
            ("preprocessor", ColumnTransformer(transformers=transformers, remainder="drop")),
            ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
        ]
    )


def train_candidate_models(df: pd.DataFrame) -> Dict[str, Any]:
    """Train a small baseline set and compare them on a stratified holdout set."""
    X, y = prepare_feature_matrix(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    models: Dict[str, Any] = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        "decision_tree": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
        "random_forest": RandomForestClassifier(random_state=42, class_weight="balanced", n_estimators=200),
    }

    results = {}
    for name, model in models.items():
        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    ColumnTransformer(
                        transformers=[
                            (
                                "numeric",
                                Pipeline(
                                    steps=[
                                        ("imputer", SimpleImputer(strategy="median")),
                                        ("scaler", StandardScaler()),
                                    ]
                                ),
                                list(X.select_dtypes(exclude=["object", "category", "bool"]).columns),
                            ),
                            (
                                "categorical",
                                Pipeline(
                                    steps=[
                                        ("imputer", SimpleImputer(strategy="most_frequent")),
                                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                                    ]
                                ),
                                list(X.select_dtypes(include=["object", "category", "bool"]).columns),
                            ),
                        ],
                        remainder="drop",
                    ),
                ),
                ("classifier", model),
            ]
        )
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        results[name] = {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred, pos_label="Yes", zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred, pos_label="Yes", zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, y_pred, pos_label="Yes", zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test.map({"Yes": 1, "No": 0}), y_prob)), 4),
            "confusion_matrix": confusion_matrix(y_test, y_pred, labels=["No", "Yes"]).tolist(),
        }

    return {
        "model_comparison": results,
        "train_shape": X_train.shape,
        "test_shape": X_test.shape,
        "positive_rate_test": round(float(y_test.eq("Yes").mean()), 4),
    }
