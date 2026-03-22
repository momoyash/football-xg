from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    brier_score_loss,
    classification_report,
    roc_auc_score,
)


def evaluate_xg_model(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray | pd.Series,
    n_bins: int = 10,
) -> dict:
    """
    Compute evaluation metrics for an xG model.

    Returns
    -------
    dict with keys:
        roc_auc        – area under the ROC curve
        brier_score    – Brier score (lower = better)
        log_loss       – binary cross-entropy
        ece            – expected calibration error
        fraction_pos   – observed goal rate per calibration bin
        mean_pred      – mean predicted xG per calibration bin
    """
    y_true = np.asarray(y_true, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)

    roc_auc = roc_auc_score(y_true, y_proba)
    brier = brier_score_loss(y_true, y_proba)

    # Log-loss (manual to avoid extra dependency)
    eps = 1e-15
    y_clip = np.clip(y_proba, eps, 1 - eps)
    log_loss = -np.mean(y_true * np.log(y_clip) + (1 - y_true) * np.log(1 - y_clip))

    # Calibration curve
    fraction_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy="uniform")

    # Expected calibration error — bin counts must match calibration curve bins
    n_cal_bins = len(fraction_pos)
    bin_sizes = np.histogram(y_proba, bins=n_cal_bins, range=(0, 1))[0]
    ece = float(np.sum(np.abs(fraction_pos - mean_pred) * bin_sizes) / len(y_true))

    return {
        "roc_auc": round(float(roc_auc), 4),
        "brier_score": round(float(brier), 4),
        "log_loss": round(float(log_loss), 4),
        "ece": round(ece, 4),
        "fraction_pos": fraction_pos,
        "mean_pred": mean_pred,
    }


def evaluate_outcome_model(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    labels: list[str] | None = None,
) -> dict:
    """
    Compute evaluation metrics for a match outcome classifier.

    Returns
    -------
    dict with keys:
        accuracy           – overall accuracy
        classification_report – full sklearn report string
        per_class          – dict of precision/recall/f1 per class
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    accuracy = float(np.mean(y_true == y_pred))
    report_str = classification_report(y_true, y_pred, labels=labels, zero_division=0)

    report_dict = classification_report(
        y_true, y_pred, labels=labels, zero_division=0, output_dict=True
    )
    per_class = {
        k: v
        for k, v in report_dict.items()
        if isinstance(v, dict)
    }

    return {
        "accuracy": round(accuracy, 4),
        "classification_report": report_str,
        "per_class": per_class,
    }


def xg_vs_actual(
    shots: pd.DataFrame,
    xg_col: str = "xg_pred",
    goal_col: str = "is_goal",
    group_col: str | None = None,
) -> pd.DataFrame:
    """
    Compare predicted xG against actual goals, optionally grouped by a column
    (e.g. team, player, match_id).

    Returns a DataFrame with columns: actual_goals, predicted_xg, difference.
    """
    shots = shots.copy()
    if group_col and group_col in shots.columns:
        result = shots.groupby(group_col).agg(
            actual_goals=(goal_col, "sum"),
            predicted_xg=(xg_col, "sum"),
        )
    else:
        result = pd.DataFrame(
            {
                "actual_goals": [shots[goal_col].sum()],
                "predicted_xg": [shots[xg_col].sum()],
            }
        )

    result["difference"] = result["actual_goals"] - result["predicted_xg"]
    return result.round(3)
