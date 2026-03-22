from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def _get_col(df: pd.DataFrame, primary: str, fallback: str) -> str:
    """
    Return the name of the first column that exists in the DataFrame.
    """
    return primary if primary in df.columns else fallback


def _ensure_time_minutes(events: pd.DataFrame) -> pd.Series:
    """
    Approximate event time in minutes from StatsBomb `minute` and `second` columns.
    """
    minute = pd.to_numeric(events.get("minute", 0), errors="coerce").fillna(0)
    second = pd.to_numeric(events.get("second", 0), errors="coerce").fillna(0)
    return minute + second / 60.0


def compute_team_possession(events: pd.DataFrame) -> pd.Series:
    """
    Compute possession percentage per team.

    If `possession_team` is present, possession is based on the sum of
    event durations per possession team. Otherwise, it falls back to the
    share of total events per `team`/`team_name`.
    """
    if "possession_team" in events.columns:
        durations = pd.to_numeric(events.get("duration", 0), errors="coerce").fillna(0.0)
        poss_time = durations.groupby(events["possession_team"]).sum()
        total_time = poss_time.sum()
        if total_time > 0:
            possession_pct = poss_time / total_time * 100.0
        else:
            possession_pct = poss_time * 0.0
    else:
        team_col = _get_col(events, "team", "team_name")
        event_counts = events.groupby(team_col).size()
        total_events = event_counts.sum()
        possession_pct = event_counts / total_events * 100.0

    possession_pct.name = "possession_pct"
    return possession_pct


def compute_passes_and_pressures_per_minute(events: pd.DataFrame) -> pd.DataFrame:
    """
    Compute passes per minute and pressures per minute for each team.

    Minutes are approximated from the maximum event time in the match.
    """
    type_col = _get_col(events, "type", "type_name")
    team_col = _get_col(events, "team", "team_name")

    time_min = _ensure_time_minutes(events)
    total_minutes = float(time_min.max() + 1e-6)  # avoid division by zero

    passes = events[events[type_col].astype(str).str.contains("Pass", na=False)]
    pressures = events[events[type_col].astype(str).str.contains("Pressure", na=False)]

    passes_per_team = passes.groupby(team_col).size().rename("passes")
    pressures_per_team = pressures.groupby(team_col).size().rename("pressures")

    features = pd.concat([passes_per_team, pressures_per_team], axis=1).fillna(0)
    features["passes_per_min"] = features["passes"] / total_minutes
    features["pressures_per_min"] = features["pressures"] / total_minutes

    return features[["passes_per_min", "pressures_per_min"]]


def _compute_shot_distance(events: pd.DataFrame) -> pd.Series:
    """
    Compute shot distance from StatsBomb coordinates.

    Assumes the origin is at the bottom-left and the goal is centered at (120, 40).
    Coordinates are taken from `location` if present; otherwise this returns NaNs.
    """
    if "location" not in events.columns:
        return pd.Series(np.nan, index=events.index, name="shot_distance")

    loc = events["location"]

    # Handle both list-like and stringified lists
    def _to_xy(v):
        if isinstance(v, (list, tuple)) and len(v) >= 2:
            return float(v[0]), float(v[1])
        if isinstance(v, str) and v.startswith("[") and v.endswith("]"):
            try:
                x_str, y_str = v.strip("[]").split(",")[:2]
                return float(x_str), float(y_str)
            except Exception:
                return np.nan, np.nan
        return np.nan, np.nan

    xy = loc.map(_to_xy)
    xs = np.array([p[0] for p in xy])
    ys = np.array([p[1] for p in xy])

    goal_x, goal_y = 120.0, 40.0
    dist = np.sqrt((goal_x - xs) ** 2 + (goal_y - ys) ** 2)

    return pd.Series(dist, index=events.index, name="shot_distance")


def compute_shot_features(events: pd.DataFrame) -> pd.DataFrame:
    """
    Compute average shot distance and total xG per team.
    """
    type_col = _get_col(events, "type", "type_name")
    team_col = _get_col(events, "team", "team_name")

    shots = events[events[type_col].astype(str).str.contains("Shot", na=False)].copy()
    if shots.empty:
        return pd.DataFrame(columns=["avg_shot_distance", "total_xg"])

    if "shot_statsbomb_xg" not in shots.columns:
        shots["shot_statsbomb_xg"] = np.nan

    shots["shot_distance"] = _compute_shot_distance(shots)

    grouped = shots.groupby(team_col).agg(
        avg_shot_distance=("shot_distance", "mean"),
        total_xg=("shot_statsbomb_xg", "sum"),
    )

    return grouped


def compute_team_features(events: pd.DataFrame) -> pd.DataFrame:
    """
    High-level helper to compute a set of team-level features:

    - possession percentage
    - passes per minute
    - pressures per minute
    - average shot distance
    - total xG

    Returns a DataFrame indexed by team name with one row per team.
    """
    team_col = _get_col(events, "team", "team_name")

    poss = compute_team_possession(events)
    tempo = compute_passes_and_pressures_per_minute(events)
    shots = compute_shot_features(events)

    features = pd.DataFrame(index=sorted(events[team_col].dropna().unique()))
    features = features.join(poss, how="left")
    features = features.join(tempo, how="left")
    features = features.join(shots, how="left")

    return features


def build_shot_dataset(events: pd.DataFrame) -> pd.DataFrame:
    """
    Build a shot-level dataset suitable for training an xG model.

    Returns a DataFrame where each row is a shot with:
    - x, y: shot coordinates (if available)
    - shot_distance: distance to centre of goal
    - minute, second: game time
    - shot_body_part, shot_technique, shot_type: categorical descriptors (if present)
    - is_goal: binary target (1 if goal, 0 otherwise)
    """
    type_col = _get_col(events, "type", "type_name")

    shots = events[events[type_col].astype(str).str.contains("Shot", na=False)].copy()
    if shots.empty:
        return pd.DataFrame()

    # Outcome / target
    outcome_col = None
    for col in ("shot_outcome", "shot_outcome_name"):
        if col in shots.columns:
            outcome_col = col
            break

    if outcome_col is not None:
        is_goal = shots[outcome_col].astype(str).str.contains("Goal", na=False)
    else:
        is_goal = pd.Series(False, index=shots.index)

    shots["is_goal"] = is_goal.astype(int)

    # Coordinates, distance
    x_vals = pd.Series(np.nan, index=shots.index, name="x")
    y_vals = pd.Series(np.nan, index=shots.index, name="y")

    if "location" in shots.columns:
        def _to_xy(v):
            if isinstance(v, (list, tuple)) and len(v) >= 2:
                return float(v[0]), float(v[1])
            if isinstance(v, str) and v.startswith("[") and v.endswith("]"):
                try:
                    x_str, y_str = v.strip("[]").split(",")[:2]
                    return float(x_str), float(y_str)
                except Exception:
                    return np.nan, np.nan
            return np.nan, np.nan

        xy = shots["location"].map(_to_xy)
        x_vals = pd.Series([p[0] for p in xy], index=shots.index, name="x")
        y_vals = pd.Series([p[1] for p in xy], index=shots.index, name="y")

    shots["x"] = x_vals
    shots["y"] = y_vals
    shots["shot_distance"] = _compute_shot_distance(shots)

    # Time features
    shots["minute"] = pd.to_numeric(shots.get("minute", 0), errors="coerce").fillna(0)
    shots["second"] = pd.to_numeric(shots.get("second", 0), errors="coerce").fillna(0)

    # Keep a compact set of columns for modeling
    cols = [
        "x",
        "y",
        "shot_distance",
        "minute",
        "second",
        "shot_body_part",
        "shot_technique",
        "shot_type",
        "is_goal",
    ]

    existing_cols = [c for c in cols if c in shots.columns]
    return shots[existing_cols].reset_index(drop=True)
