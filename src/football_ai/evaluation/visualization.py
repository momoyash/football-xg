from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mplsoccer import Pitch


def plot_shot_map(
    shots: pd.DataFrame,
    x_col: str = "x",
    y_col: str = "y",
    xg_col: str | None = "shot_statsbomb_xg",
    goal_col: str | None = "is_goal",
    team_col: str | None = "team",
    title: str = "Shot Map",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Draw a shot map on a StatsBomb pitch.

    Marker size scales with xG (if available). Goals are shown as filled
    circles; non-goals as hollow circles.
    """
    pitch = Pitch(pitch_type="statsbomb", pitch_color="grass", line_color="white")
    fig, ax = pitch.draw(figsize=(12, 8))

    teams = shots[team_col].unique() if team_col and team_col in shots.columns else [None]
    colors = plt.cm.Set1(np.linspace(0, 0.7, len(teams)))

    for team, color in zip(teams, colors):
        subset = shots[shots[team_col] == team] if team is not None else shots

        x = subset[x_col].to_numpy(dtype=float) if x_col in subset.columns else np.full(len(subset), np.nan)
        y = subset[y_col].to_numpy(dtype=float) if y_col in subset.columns else np.full(len(subset), np.nan)
        xg = subset[xg_col].fillna(0.05).to_numpy(dtype=float) if xg_col and xg_col in subset.columns else np.full(len(subset), 0.05)
        is_goal = subset[goal_col].astype(bool).to_numpy() if goal_col and goal_col in subset.columns else np.zeros(len(subset), dtype=bool)

        sizes = (xg * 1000).clip(50, 1000)

        # Non-goals
        pitch.scatter(x[~is_goal], y[~is_goal], s=sizes[~is_goal],
                      edgecolors=color, facecolors="none", linewidths=1.5,
                      ax=ax, label=f"{team} shot" if team else "Shot")
        # Goals
        pitch.scatter(x[is_goal], y[is_goal], s=sizes[is_goal],
                      edgecolors=color, facecolors=color, linewidths=1.5,
                      ax=ax, label=f"{team} goal" if team else "Goal", zorder=5)

    ax.legend(loc="upper left", fontsize=9)
    ax.set_title(title, fontsize=14, pad=12)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_xg_calibration(
    fraction_pos: Sequence[float],
    mean_pred: Sequence[float],
    title: str = "xG Calibration",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot a calibration curve comparing predicted xG to observed goal rates.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(mean_pred, fraction_pos, "o-", color="steelblue", label="Model")

    ax.set_xlabel("Mean predicted xG")
    ax.set_ylabel("Fraction of goals")
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_feature_importance(
    feature_names: Sequence[str],
    importances: Sequence[float],
    top_n: int = 20,
    title: str = "Feature Importance",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Horizontal bar chart of feature importances.
    """
    names = list(feature_names)
    imps = np.asarray(importances, dtype=float)

    order = np.argsort(imps)[::-1][:top_n]
    names = [names[i] for i in order]
    imps = imps[order]

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
    ax.barh(range(len(names)), imps[::-1], color="steelblue")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1], fontsize=9)
    ax.set_xlabel("Importance")
    ax.set_title(title)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_team_features(
    team_features: pd.DataFrame,
    metrics: Sequence[str] | None = None,
    title: str = "Team Feature Comparison",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Radar / bar chart comparing team-level features across teams.

    Uses a grouped bar chart when there are more than 2 teams; a simple
    side-by-side bar chart for exactly 2 teams.
    """
    if metrics is None:
        metrics = [c for c in team_features.columns if c not in ("team", "match_id", "result")]

    data = team_features.set_index("team")[metrics] if "team" in team_features.columns else team_features[metrics]
    data = data.apply(pd.to_numeric, errors="coerce").fillna(0)

    fig, ax = plt.subplots(figsize=(max(8, len(metrics) * 1.5), 5))
    x = np.arange(len(metrics))
    n_teams = len(data)
    width = 0.8 / n_teams

    colors = plt.cm.Set2(np.linspace(0, 1, n_teams))
    for i, (team, row) in enumerate(data.iterrows()):
        ax.bar(x + i * width, row[metrics].values, width=width,
               label=str(team), color=colors[i])

    ax.set_xticks(x + width * (n_teams - 1) / 2)
    ax.set_xticklabels(metrics, rotation=25, ha="right", fontsize=9)
    ax.legend(fontsize=9)
    ax.set_title(title)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
