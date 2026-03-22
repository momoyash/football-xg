from __future__ import annotations

from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


def load_team_features(
    csv_path: str | Path,
    target_column: str = "result",
    feature_columns: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load team-level features and outcome labels from a CSV file.

    The CSV is expected to have one row per team per match, with:
    - numeric feature columns (e.g. possession_pct, passes_per_min, total_xg, ...)
    - a target column indicating match outcome from that team's perspective,
      e.g. 'win', 'draw', 'loss'.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in {csv_path}")

    y = df[target_column].astype(str)

    if feature_columns is None:
        feature_columns = [
            c
            for c in df.columns
            if c not in {target_column, "team", "team_name", "match_id"}
        ]

    X = df[list(feature_columns)].copy()

    # Ensure numeric features
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    return X, y


def train_match_outcome_model(
    features_csv: str | Path,
    model_out: str | Path = "models/artifacts/match_outcome_model.joblib",
    target_column: str = "result",
    feature_columns: Sequence[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Path:
    """
    Train a scikit-learn model to predict match outcome from team features.

    Parameters
    ----------
    features_csv : str | Path
        Path to a CSV containing team-level features and an outcome column.
    model_out : str | Path, default 'models/artifacts/match_outcome_model.joblib'
        Where to save the trained model.
    target_column : str, default 'result'
        Name of the outcome column in the CSV (e.g. 'result' with values win/draw/loss).
    feature_columns : sequence of str, optional
        Subset of columns to use as features. If None, all non-target, non-ID columns are used.
    test_size : float, default 0.2
        Fraction of data to use for the validation set.
    random_state : int, default 42
        Random seed for train/validation split and model.
    """
    X, y = load_team_features(
        csv_path=features_csv,
        target_column=target_column,
        feature_columns=feature_columns,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Pipeline: standardize features then fit a RandomForest classifier
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_val)
    report = classification_report(y_val, y_pred, digits=3)
    print("Validation classification report:")
    print(report)

    model_out = Path(model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_out)

    print(f"Saved trained model to: {model_out.resolve()}")
    return model_out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train a scikit-learn model to predict match outcome from team features."
    )
    parser.add_argument(
        "features_csv",
        type=str,
        help="Path to CSV with team-level features and a 'result' column.",
    )
    parser.add_argument(
        "--model-out",
        type=str,
        default="models/artifacts/match_outcome_model.joblib",
        help="Path to save the trained model (joblib).",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default="result",
        help="Name of the target column in the CSV (default: result).",
    )

    args = parser.parse_args()

    train_match_outcome_model(
        features_csv=args.features_csv,
        model_out=args.model_out,
        target_column=args.target_column,
    )
