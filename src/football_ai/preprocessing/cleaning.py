from __future__ import annotations

import ast
from typing import List

import numpy as np
import pandas as pd


def drop_duplicate_events(events: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with duplicate StatsBomb event IDs."""
    if "id" in events.columns:
        return events.drop_duplicates(subset=["id"]).reset_index(drop=True)
    return events.drop_duplicates().reset_index(drop=True)


def parse_location_column(events: pd.DataFrame, col: str = "location") -> pd.DataFrame:
    """
    Parse a stringified-list location column (e.g. '[60.3, 40.1]') into
    separate float columns ``{col}_x`` and ``{col}_y``.
    """
    if col not in events.columns:
        return events

    def _parse(v):
        if isinstance(v, (list, tuple)) and len(v) >= 2:
            return float(v[0]), float(v[1])
        if isinstance(v, str):
            try:
                parsed = ast.literal_eval(v)
                if isinstance(parsed, (list, tuple)) and len(parsed) >= 2:
                    return float(parsed[0]), float(parsed[1])
            except Exception:
                pass
        return np.nan, np.nan

    xy = events[col].map(_parse)
    events = events.copy()
    events[f"{col}_x"] = [p[0] for p in xy]
    events[f"{col}_y"] = [p[1] for p in xy]
    return events


def normalize_column_names(events: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names: lowercase, strip whitespace, replace spaces with
    underscores. Also maps legacy ``type_name`` / ``team_name`` aliases to
    ``type`` / ``team`` if the canonical names are absent.
    """
    events = events.copy()
    events.columns = [c.strip().lower().replace(" ", "_") for c in events.columns]

    aliases = {"type_name": "type", "team_name": "team", "player_name": "player"}
    for old, new in aliases.items():
        if old in events.columns and new not in events.columns:
            events = events.rename(columns={old: new})

    return events


def fill_missing_numeric(
    events: pd.DataFrame,
    columns: List[str] | None = None,
    fill_value: float = 0.0,
) -> pd.DataFrame:
    """
    Fill NaN values in numeric columns with *fill_value* (default 0.0).
    If *columns* is None, all numeric columns are filled.
    """
    events = events.copy()
    if columns is None:
        columns = events.select_dtypes(include="number").columns.tolist()
    for col in columns:
        if col in events.columns:
            events[col] = pd.to_numeric(events[col], errors="coerce").fillna(fill_value)
    return events


def filter_event_types(events: pd.DataFrame, include: List[str]) -> pd.DataFrame:
    """
    Keep only rows whose ``type`` column matches one of the *include* values
    (case-insensitive substring match).

    Example
    -------
    >>> shots = filter_event_types(events, include=["Shot"])
    """
    type_col = "type" if "type" in events.columns else "type_name"
    if type_col not in events.columns:
        return events

    pattern = "|".join(include)
    mask = events[type_col].astype(str).str.contains(pattern, case=False, na=False)
    return events[mask].reset_index(drop=True)


def clean_events(events: pd.DataFrame) -> pd.DataFrame:
    """
    Apply a standard cleaning pipeline to a raw StatsBomb events DataFrame:

    1. Normalize column names
    2. Drop duplicate events
    3. Parse ``location``, ``pass_end_location``, ``carry_end_location`` columns
    4. Fill missing numerics with 0
    """
    events = normalize_column_names(events)
    events = drop_duplicate_events(events)
    for loc_col in ("location", "pass_end_location", "carry_end_location"):
        events = parse_location_column(events, col=loc_col)
    events = fill_missing_numeric(events)
    return events
