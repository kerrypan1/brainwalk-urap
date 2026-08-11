"""Sklearn preprocessors for numeric features (fold-internal fit only)."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features.columns import reject_leaky_columns


def build_numeric_preprocessor() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def build_model_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    reject_leaky_columns(list(X.columns), context="model preprocessor")
    return ColumnTransformer(
        transformers=[
            ("num", build_numeric_preprocessor(), list(X.columns)),
        ],
        remainder="drop",
    )
