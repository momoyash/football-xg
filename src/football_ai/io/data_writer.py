from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd


def save_dataframe(df: pd.DataFrame, path: str | Path, index: bool = False) -> Path:
    """
    Save a DataFrame to CSV, creating parent directories as needed.
    Returns the resolved output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    return path.resolve()


def save_model(model: object, path: str | Path) -> Path:
    """
    Persist a scikit-learn pipeline / model with joblib.
    Returns the resolved output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path.resolve()


def load_model(path: str | Path) -> object:
    """Load a joblib-serialised model from disk."""
    return joblib.load(Path(path))


def save_predictions(
    shots: pd.DataFrame,
    xg_preds: "pd.Series | None" = None,
    path: str | Path = "data/processed/shot_predictions.csv",
) -> Path:
    """
    Attach xG predictions to a shot DataFrame and write to CSV.

    If *xg_preds* is None the DataFrame is written as-is (useful when
    predictions are already a column).
    """
    out = shots.copy()
    if xg_preds is not None:
        out["xg_pred"] = xg_preds.values if hasattr(xg_preds, "values") else xg_preds
    return save_dataframe(out, path)


def save_match_report(stats: dict, path: str | Path) -> Path:
    """
    Write a text match report from a ``compute_match_stats`` result dict.
    """
    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "=== Match Report ===",
        f"Total passes   : {stats.get('total_passes', 'N/A')}",
        f"Total shots    : {stats.get('total_shots', 'N/A')}",
        f"Total pressures: {stats.get('total_pressures', 'N/A')}",
    ]
    xg_val = stats.get("total_xg", float("nan"))
    if not (isinstance(xg_val, float) and np.isnan(xg_val)):
        lines.append(f"Total xG       : {xg_val:.3f}")
    else:
        lines.append("Total xG       : N/A")

    per_team: pd.DataFrame | None = stats.get("per_team_stats")
    if per_team is not None:
        lines.append("\n=== Per-team ===")
        lines.append(per_team.to_string())

    poss: pd.Series | None = stats.get("possession_pct")
    if poss is not None:
        lines.append("\n=== Possession ===")
        for team, pct in poss.items():
            lines.append(f"  {team}: {pct:.1f}%")

    text = "\n".join(lines)
    path.write_text(text, encoding="utf-8")
    return path.resolve()
