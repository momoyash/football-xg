from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from football_ai.preprocessing.feature_engineering import build_shot_dataset, compute_team_features


# ---------------------------------------------------------------------------
# Shot-level dataset
# ---------------------------------------------------------------------------

def load_shot_dataset(
    csv_files: Iterable[str | Path],
    drop_missing_coords: bool = True,
) -> pd.DataFrame:
    """
    Load and concatenate shot-level datasets from multiple StatsBomb event CSVs.

    Parameters
    ----------
    csv_files :
        Iterable of paths to StatsBomb event CSVs.
    drop_missing_coords :
        If True, drop rows where x or y coordinates are NaN.

    Returns
    -------
    DataFrame with one row per shot and columns including x, y,
    shot_distance, minute, second, categorical shot descriptors, and is_goal.
    """
    frames: list[pd.DataFrame] = []
    for p in csv_files:
        events = pd.read_csv(p)
        shots = build_shot_dataset(events)
        if not shots.empty:
            frames.append(shots)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    if drop_missing_coords:
        combined = combined.dropna(subset=["x", "y"]).reset_index(drop=True)

    return combined


def load_team_features_dataset(
    csv_files: Iterable[str | Path],
    competition_id: int | None = None,
    season_id: int | None = None,
    result_map: dict[int, dict[str, str]] | None = None,
) -> pd.DataFrame:
    """
    Compute team-level features from multiple StatsBomb event CSVs and
    concatenate into one DataFrame.

    If *result_map* is supplied (match_id -> {team: result}), a ``result``
    column is added. Otherwise only features are returned.

    Parameters
    ----------
    csv_files :
        Iterable of paths to events CSVs.
    result_map :
        Optional mapping of match_id to per-team results ('win'/'draw'/'loss').
        Typically obtained from ``statsbombpy.sb.matches()``.
    """
    frames: list[pd.DataFrame] = []
    for p in csv_files:
        p = Path(p)
        match_id = int(p.stem.split("_")[1]) if "_" in p.stem else -1
        events = pd.read_csv(p)
        feats = compute_team_features(events)
        feats = feats.reset_index().rename(columns={"index": "team"})
        feats["match_id"] = match_id

        if result_map and match_id in result_map:
            feats["result"] = feats["team"].map(result_map[match_id])

        frames.append(feats)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# X / y splitter helpers
# ---------------------------------------------------------------------------

def split_shots_xy(
    shots: pd.DataFrame,
    target_col: str = "is_goal",
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split a shot DataFrame into feature matrix X and target series y.

    Default numeric features: x, y, shot_distance, minute, second
    Default categorical features: shot_body_part, shot_technique, shot_type
    """
    if numeric_features is None:
        numeric_features = ["x", "y", "shot_distance", "minute", "second"]
    if categorical_features is None:
        categorical_features = ["shot_body_part", "shot_technique", "shot_type"]

    feature_cols = [c for c in numeric_features + categorical_features if c in shots.columns]
    X = shots[feature_cols].copy()
    y = shots[target_col].astype(int)
    return X, y


def split_team_features_xy(
    team_features: pd.DataFrame,
    target_col: str = "result",
    drop_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split a team features DataFrame into X and y, dropping ID / label columns.
    """
    if drop_cols is None:
        drop_cols = ["team", "team_name", "match_id", target_col]

    feature_cols = [c for c in team_features.columns if c not in drop_cols]
    X = team_features[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = team_features[target_col].astype(str)
    return X, y
