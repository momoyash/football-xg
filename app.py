"""
Football Analytics AI — Streamlit Dashboard
Run with: streamlit run app.py
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from statsbombpy import sb

from football_ai.modeling.datasets import load_shot_dataset, split_shots_xy, split_team_features_xy
from football_ai.modeling.models import get_model
from football_ai.evaluation.metrics import evaluate_xg_model, evaluate_outcome_model
from football_ai.evaluation.visualization import (
    plot_shot_map,
    plot_xg_calibration,
    plot_feature_importance,
    plot_team_features,
)
from football_ai.preprocessing.feature_engineering import build_shot_dataset, _compute_shot_distance

# ── constants ────────────────────────────────────────────────────────────────
COMP_ID = 43
SEASON_ID = 3
RAW_DIR = Path("data/raw/comp_43_season_3")
TEAM_FEATURES_CSV = Path("data/team_features.csv")

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Football Analytics AI",
    page_icon="⚽",
    layout="wide",
)

st.title("⚽ Football Analytics AI")
st.caption("2018 FIFA World Cup · StatsBomb Open Data · xG & Match Outcome Models")

# ── sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Shot Map", "xG Model", "Team Stats", "Match Outcome Model"],
)

# ── cached data loaders ───────────────────────────────────────────────────────

@st.cache_data
def load_matches():
    return sb.matches(competition_id=COMP_ID, season_id=SEASON_ID)

@st.cache_data
def load_team_features():
    return pd.read_csv(TEAM_FEATURES_CSV)

@st.cache_data
def get_csv_files():
    return sorted(RAW_DIR.glob("events_*.csv"))

@st.cache_data
def get_shot_dataset():
    csvs = get_csv_files()
    return load_shot_dataset(csvs)

@st.cache_data
def train_xg_model_cached():
    shots = get_shot_dataset()
    X, y = split_shots_xy(shots)
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = get_model("xg_gbm")
    model.fit(X_tr, y_tr)
    y_proba = model.predict_proba(X_val)[:, 1]
    metrics = evaluate_xg_model(y_val, y_proba)
    return model, X_val, y_val, y_proba, metrics

@st.cache_data
def train_outcome_model_cached():
    team_df = load_team_features()
    X, y = split_team_features_xy(team_df)
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = get_model("outcome_rf")
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_val)
    metrics = evaluate_outcome_model(y_val, y_pred)
    return model, metrics


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: Overview
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.header("Project Overview")

    col1, col2, col3, col4 = st.columns(4)
    matches = load_matches()
    csvs = get_csv_files()
    shots = get_shot_dataset()
    team_df = load_team_features()

    col1.metric("Matches", len(matches))
    col2.metric("Teams", len(set(matches["home_team"]) | set(matches["away_team"])))
    col3.metric("Total Shots", len(shots))
    col4.metric("Goal Rate", f"{shots['is_goal'].mean():.1%}")

    st.markdown("---")

    st.subheader("What this project does")
    st.markdown("""
This is a **football analytics pipeline** built on StatsBomb open data from the **2018 FIFA World Cup**.

It answers two core questions using machine learning:

1. **xG Model** — *How likely was each shot to become a goal?*
   - Uses shot location, distance, body part, and technique
   - Output: a probability (0–1) called **expected goals (xG)**

2. **Match Outcome Model** — *Can we predict who wins from team-level stats?*
   - Features: possession %, passes/min, pressures/min, average shot distance, total xG
   - Output: win / draw / loss prediction

**Data source:** StatsBomb Open Data — all 64 matches, 32 teams, ~200,000 events.
    """)

    st.markdown("---")
    st.subheader("Tournament xG Leaderboard")
    xg_total = team_df.groupby("team")["total_xg"].sum().sort_values(ascending=False)
    st.bar_chart(xg_total)

    st.subheader("Match Results Distribution")
    result_counts = team_df["result"].value_counts()
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(result_counts.values, labels=result_counts.index,
           autopct="%1.0f%%", colors=["#2a9d8f", "#e9c46a", "#e76f51"])
    ax.set_title("Match Results (team perspective)")
    st.pyplot(fig)
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: Shot Map
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Shot Map":
    st.header("Shot Map")

    matches = load_matches()
    match_options = {
        f"{row['home_team']} vs {row['away_team']} (Week {row['match_week']})": int(row["match_id"])
        for _, row in matches.sort_values("match_week").iterrows()
    }

    selected = st.selectbox("Select a match", list(match_options.keys()))
    match_id = match_options[selected]

    csv_path = RAW_DIR / f"events_{match_id}.csv"
    if csv_path.exists():
        events = pd.read_csv(csv_path)
        shot_rows = events[events["type"].astype(str).str.contains("Shot", na=False)].copy()

        def _to_xy(v):
            if isinstance(v, (list, tuple)) and len(v) >= 2:
                return float(v[0]), float(v[1])
            if isinstance(v, str) and v.startswith("["):
                try:
                    parts = v.strip("[]").split(",")
                    return float(parts[0]), float(parts[1])
                except Exception:
                    pass
            return np.nan, np.nan

        xy = shot_rows["location"].map(_to_xy)
        shot_rows = shot_rows.copy()
        shot_rows["x"] = [p[0] for p in xy]
        shot_rows["y"] = [p[1] for p in xy]
        shot_rows["is_goal"] = shot_rows["shot_outcome"].astype(str).str.contains("Goal").astype(int)

        col1, col2 = st.columns(2)
        teams = shot_rows["team"].dropna().unique()
        for team in teams:
            t_shots = shot_rows[shot_rows["team"] == team]
            col1.metric(f"{team} shots", len(t_shots))
            col2.metric(f"{team} goals", int(t_shots["is_goal"].sum()))

        fig = plot_shot_map(
            shot_rows, x_col="x", y_col="y",
            xg_col="shot_statsbomb_xg", goal_col="is_goal", team_col="team",
            title=f"Shot Map: {selected}",
        )
        st.pyplot(fig)
        plt.close()

        st.subheader("Shot details")
        display_cols = [c for c in ["team", "player", "minute", "x", "y",
                                     "shot_statsbomb_xg", "shot_outcome",
                                     "shot_body_part", "shot_technique"] if c in shot_rows.columns]
        st.dataframe(shot_rows[display_cols].reset_index(drop=True), use_container_width=True)
    else:
        st.warning(f"CSV not found for match {match_id}.")


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: xG Model
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "xG Model":
    st.header("Expected Goals (xG) Model")

    st.markdown("""
**What is xG?**
Expected Goals is a metric that estimates the probability a shot results in a goal,
based on where it was taken from, how it was struck, and with which body part.
A value of 0.8 means the model thinks that shot had an 80% chance of being a goal.
    """)

    with st.spinner("Training xG model on all 64 matches..."):
        model, X_val, y_val, y_proba, metrics = train_xg_model_cached()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ROC-AUC", metrics["roc_auc"], help="1.0 = perfect, 0.5 = random")
    col2.metric("Brier Score", metrics["brier_score"], help="Lower = better calibrated")
    col3.metric("Log Loss", metrics["log_loss"])
    col4.metric("ECE", metrics["ece"], help="Expected Calibration Error")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Calibration Curve")
        fig = plot_xg_calibration(
            metrics["fraction_pos"], metrics["mean_pred"],
            title="Predicted xG vs Actual Goal Rate",
        )
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.subheader("Feature Importance")
        num_feats = [c for c in ["x", "y", "shot_distance", "minute", "second"] if c in X_val.columns]
        ohe = model.named_steps["preprocess"].named_transformers_["cat"]
        cat_feats = list(ohe.get_feature_names_out(["shot_body_part", "shot_technique", "shot_type"]))
        all_feats = num_feats + cat_feats
        imps = model.named_steps["clf"].feature_importances_
        fig = plot_feature_importance(all_feats, imps, top_n=15, title="Top 15 Features")
        st.pyplot(fig)
        plt.close()

    st.subheader("xG Distribution")
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.hist(y_proba[y_val == 0], bins=40, alpha=0.6, label="No goal", color="#457b9d")
    ax.hist(y_proba[y_val == 1], bins=20, alpha=0.7, label="Goal", color="#e63946")
    ax.set_xlabel("Predicted xG")
    ax.set_ylabel("Count")
    ax.set_title("xG distribution: goals vs non-goals")
    ax.legend()
    st.pyplot(fig)
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: Team Stats
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Team Stats":
    st.header("Team Statistics")

    team_df = load_team_features()

    st.subheader("Tournament Totals")
    agg = team_df.groupby("team").agg(
        matches=("match_id", "count"),
        total_xg=("total_xg", "sum"),
        avg_possession=("possession_pct", "mean"),
        avg_passes_per_min=("passes_per_min", "mean"),
        avg_pressures_per_min=("pressures_per_min", "mean"),
        avg_shot_distance=("avg_shot_distance", "mean"),
    ).round(2).sort_values("total_xg", ascending=False)

    st.dataframe(agg, use_container_width=True)

    st.subheader("Compare Teams")
    all_teams = sorted(team_df["team"].unique())
    selected_teams = st.multiselect("Select teams to compare", all_teams,
                                     default=["France", "Croatia", "Belgium", "England"])

    if selected_teams:
        subset = agg.loc[agg.index.isin(selected_teams)].reset_index()
        metric = st.selectbox("Metric", ["total_xg", "avg_possession",
                                          "avg_passes_per_min", "avg_pressures_per_min"])
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = plt.cm.Set2(np.linspace(0, 1, len(subset)))
        ax.bar(subset["team"], subset[metric], color=colors)
        ax.set_title(f"{metric} — selected teams")
        ax.set_ylabel(metric)
        st.pyplot(fig)
        plt.close()

    st.subheader("Full Feature Comparison")
    if selected_teams:
        plot_data = team_df[team_df["team"].isin(selected_teams)].groupby("team").mean(numeric_only=True).reset_index()
        fig = plot_team_features(
            plot_data,
            metrics=["possession_pct", "passes_per_min", "pressures_per_min", "total_xg"],
            title="Average Match Stats",
        )
        st.pyplot(fig)
        plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: Match Outcome Model
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Match Outcome Model":
    st.header("Match Outcome Prediction Model")

    st.markdown("""
Predicts whether a team **wins, draws, or loses** based on 5 in-match team features:
possession %, passes per minute, pressures per minute, average shot distance, and total xG.
    """)

    with st.spinner("Training outcome model..."):
        model, metrics = train_outcome_model_cached()

    st.metric("Validation Accuracy", f"{metrics['accuracy']:.1%}")

    st.subheader("Classification Report")
    st.code(metrics["classification_report"])

    st.subheader("Per-class Performance")
    per_class = {k: v for k, v in metrics["per_class"].items()
                 if k in ("win", "draw", "loss")}
    report_df = pd.DataFrame(per_class).T[["precision", "recall", "f1-score"]].round(3)
    st.dataframe(report_df, use_container_width=True)

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    for ax, (cls, vals) in zip(axes, per_class.items()):
        bars = ax.bar(["Precision", "Recall", "F1"],
                      [vals["precision"], vals["recall"], vals["f1-score"]],
                      color=["#2a9d8f", "#e9c46a", "#e76f51"])
        ax.set_ylim(0, 1)
        ax.set_title(cls.upper())
        for bar, val in zip(bars, [vals["precision"], vals["recall"], vals["f1-score"]]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.2f}", ha="center", fontsize=9)
    plt.suptitle("Precision / Recall / F1 per outcome class", y=1.02)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("Feature Importances")
    feat_names = ["possession_pct", "passes_per_min", "pressures_per_min",
                  "avg_shot_distance", "total_xg"]
    imps = model.named_steps["clf"].feature_importances_
    fig = plot_feature_importance(feat_names, imps, top_n=5,
                                   title="Which features predict match outcome?")
    st.pyplot(fig)
    plt.close()
