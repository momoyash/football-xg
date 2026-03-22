from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

from football_ai.preprocessing.feature_engineering import build_shot_dataset


def load_shots_from_csvs(csv_files: Iterable[Path]) -> pd.DataFrame:
    """
    Load multiple StatsBomb event CSVs and build a combined shot-level dataset.
    """
    dfs: list[pd.DataFrame] = []
    for p in csv_files:
        df = pd.read_csv(p)
        shots = build_shot_dataset(df)
        if not shots.empty:
            dfs.append(shots)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def train_xg_pipeline_from_events(
    events_glob: str = "data/raw/**/events_*.csv",
    model_out: str | Path = "models/artifacts/xg_model.joblib",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Path:
    """
    Train a simple expected goals (xG) model from StatsBomb shot events.

    The model predicts the probability that a shot becomes a goal (is_goal),
    using spatial and contextual shot features. The predicted probability can
    be interpreted as xG.
    """
    root = Path(".")
    csv_files = list(root.glob(events_glob))
    if not csv_files:
        raise FileNotFoundError(f"No event CSV files found for pattern: {events_glob}")

    shots = load_shots_from_csvs(csv_files)
    if shots.empty:
        raise ValueError("No shots found in the provided event files.")

    y = shots["is_goal"].astype(int).values

    numeric_features = [c for c in ["x", "y", "shot_distance", "minute", "second"] if c in shots.columns]
    categorical_features = [
        c for c in ["shot_body_part", "shot_technique", "shot_type"] if c in shots.columns
    ]

    X = shots[numeric_features + categorical_features].copy()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    clf = GradientBoostingClassifier(
        random_state=random_state,
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", clf),
        ]
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_proba)
    brier = brier_score_loss(y_val, y_proba)
    print(f"Validation ROC-AUC: {auc:.3f}")
    print(f"Validation Brier score: {brier:.3f}")

    model_out = Path(model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_out)
    print(f"Saved xG model to: {model_out.resolve()}")

    return model_out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train an expected goals (xG) model from StatsBomb shot events."
    )
    parser.add_argument(
        "--events-glob",
        type=str,
        default="data/raw/**/events_*.csv",
        help="Glob pattern for StatsBomb event CSVs (default: data/raw/**/events_*.csv).",
    )
    parser.add_argument(
        "--model-out",
        type=str,
        default="models/artifacts/xg_model.joblib",
        help="Path to save the trained xG model (joblib).",
    )

    args = parser.parse_args()

    train_xg_pipeline_from_events(
        events_glob=args.events_glob,
        model_out=args.model_out,
    )

