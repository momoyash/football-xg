from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def load_events(csv_path: str | Path) -> pd.DataFrame:
    """
    Load a single StatsBomb events CSV into a DataFrame.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Events CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    return df


def compute_match_stats(events: pd.DataFrame) -> dict:
    """
    Compute basic match-level statistics from a StatsBomb events DataFrame.

    Returns a dictionary with overall counts and per-team stats.
    """
    stats: dict[str, object] = {}

    # Column names can vary slightly between exports
    type_col = "type" if "type" in events.columns else "type_name"
    team_col = "team" if "team" in events.columns else "team_name"

    # Basic filters
    passes = events[events[type_col].astype(str).str.contains("Pass", na=False)]
    shots = events[events[type_col].astype(str).str.contains("Shot", na=False)]
    pressures = events[events[type_col].astype(str).str.contains("Pressure", na=False)]

    stats["total_passes"] = int(len(passes))
    stats["total_shots"] = int(len(shots))
    stats["total_pressures"] = int(len(pressures))

    # xG
    xg_col = "shot_statsbomb_xg" if "shot_statsbomb_xg" in shots.columns else None
    if xg_col is None:
        total_xg = float("nan")
    else:
        total_xg = float(shots[xg_col].fillna(0).sum())
    stats["total_xg"] = total_xg

    # Per-team breakdown for passes, shots, pressures, xG
    team_group = {
        "passes": passes.groupby(team_col).size(),
        "shots": shots.groupby(team_col).size(),
        "pressures": pressures.groupby(team_col).size(),
    }
    if xg_col is not None:
        team_group["xg"] = shots.groupby(team_col)[xg_col].sum()

    per_team = pd.concat(team_group, axis=1).fillna(0)
    # Ensure numeric types
    for col in per_team.columns:
        per_team[col] = pd.to_numeric(per_team[col], errors="coerce").fillna(0)

    stats["per_team_stats"] = per_team

    # Possession share per team (by duration if available, otherwise by event count)
    if "possession_team" in events.columns:
        # Use duration where available; fallback to 1.0 per event where missing
        durations = events.get("duration", pd.Series(1.0, index=events.index))
        durations = pd.to_numeric(durations, errors="coerce").fillna(0.0)
        poss_time = durations.groupby(events["possession_team"]).sum()
        total_time = poss_time.sum()
        if total_time > 0:
            possession_pct = poss_time / total_time * 100.0
        else:
            possession_pct = poss_time * 0.0
        stats["possession_pct"] = possession_pct
    else:
        # Approximate possession by share of all events per team
        event_counts = events.groupby(team_col).size()
        total_events = event_counts.sum()
        possession_pct = event_counts / total_events * 100.0
        stats["possession_pct"] = possession_pct

    return stats


def print_match_stats(stats: dict) -> None:
    """
    Pretty-print match statistics computed by `compute_match_stats`.
    """
    print("=== Overall match statistics ===")
    print(f"Total passes   : {stats['total_passes']}")
    print(f"Total shots    : {stats['total_shots']}")
    print(f"Total pressures: {stats['total_pressures']}")
    print(f"Total xG       : {stats['total_xg']:.3f}" if not np.isnan(stats["total_xg"]) else "Total xG       : N/A")

    print("\n=== Per-team statistics ===")
    per_team: pd.DataFrame = stats["per_team_stats"]
    print(per_team)

    print("\n=== Possession share (%) ===")
    possession_pct: pd.Series = stats["possession_pct"]
    for team, pct in possession_pct.items():
        print(f"{team}: {pct:.1f}%")


def summarize_events_csv(csv_path: str | Path) -> None:
    """
    Load a StatsBomb events CSV and print basic match statistics.
    """
    events = load_events(csv_path)
    stats = compute_match_stats(events)
    print_match_stats(stats)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Summarize a StatsBomb events CSV with basic match statistics."
    )
    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to a StatsBomb events CSV file (e.g. data/raw/comp_43_season_3/events_7525.csv).",
    )

    args = parser.parse_args()
    summarize_events_csv(args.csv_path)

