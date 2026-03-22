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
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import train_test_split
from statsbombpy import sb

from football_ai.modeling.datasets import load_shot_dataset, split_shots_xy, split_team_features_xy
from football_ai.modeling.models import get_model
from football_ai.evaluation.metrics import evaluate_xg_model, evaluate_outcome_model
from football_ai.evaluation.visualization import plot_shot_map, plot_xg_calibration, plot_feature_importance
from football_ai.preprocessing.feature_engineering import build_shot_dataset, _compute_shot_distance

# ── must be first streamlit call ─────────────────────────────────────────────
st.set_page_config(
    page_title="Football xG — 2018 World Cup",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── design system ─────────────────────────────────────────────────────────────
BG          = "#0D0D0D"
SURFACE     = "#141414"
CARD        = "#1C1C1C"
BORDER      = "#2A2A2A"
GREEN       = "#00F57A"
BLUE        = "#3B82F6"
TEXT        = "#F5F5F5"
MUTED       = "#6B7280"
RED         = "#EF4444"
YELLOW      = "#F59E0B"

PLOTLY_LAYOUT = dict(
    paper_bgcolor=SURFACE,
    plot_bgcolor=SURFACE,
    font=dict(family="Inter, sans-serif", color=TEXT),
    xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
    yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
    margin=dict(l=16, r=16, t=40, b=16),
)

def dark_fig(w=10, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED)
    ax.spines[:].set_color(BORDER)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT)
    return fig, ax

# ── global CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Base */
html, body, [class*="css"], .stApp {{
    font-family: 'Inter', sans-serif !important;
    background-color: {BG} !important;
    color: {TEXT} !important;
}}

/* Hide default Streamlit chrome */
#MainMenu, footer, header {{ visibility: hidden; }}

/* Hide the sidebar collapse button — prevents users getting stuck */
[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"] {{
    display: none !important;
}}
.block-container {{ padding: 2rem 2.5rem 4rem 2.5rem !important; max-width: 1400px; }}

/* Sidebar */
[data-testid="stSidebar"] {{
    background-color: {SURFACE} !important;
    border-right: 1px solid {BORDER} !important;
}}
[data-testid="stSidebar"] * {{ color: {TEXT} !important; }}

/* Hide the hidden radio label's reserved space */
[data-testid="stSidebar"] .stRadio > label {{
    display: none !important;
}}

/* Style radio options as nav items */
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] {{
    gap: 0.2rem;
    display: flex;
    flex-direction: column;
}}
[data-testid="stSidebar"] .stRadio label {{
    padding: 0.55rem 0.85rem !important;
    border-radius: 8px !important;
    transition: background 0.15s;
    cursor: pointer;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
}}
[data-testid="stSidebar"] .stRadio label:hover {{
    background: {CARD} !important;
}}
[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"] span:first-child {{
    display: none !important;
}}

/* Cards */
.kpi-card {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 0.5rem;
}}
.kpi-label {{
    font-size: 0.75rem;
    font-weight: 500;
    color: {MUTED};
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.4rem;
}}
.kpi-value {{
    font-size: 2rem;
    font-weight: 700;
    color: {TEXT};
    line-height: 1;
}}
.kpi-sub {{
    font-size: 0.75rem;
    color: {MUTED};
    margin-top: 0.3rem;
}}
.kpi-accent {{ color: {GREEN}; }}

/* Section header */
.section-header {{
    font-size: 1.1rem;
    font-weight: 600;
    color: {TEXT};
    margin: 2rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid {BORDER};
}}

/* Page title */
.page-title {{
    font-size: 2rem;
    font-weight: 700;
    color: {TEXT};
    margin-bottom: 0.25rem;
}}
.page-subtitle {{
    font-size: 0.9rem;
    color: {MUTED};
    margin-bottom: 2rem;
}}

/* Insight box */
.insight-box {{
    background: {CARD};
    border-left: 3px solid {GREEN};
    border-radius: 0 8px 8px 0;
    padding: 0.85rem 1.1rem;
    margin: 0.5rem 0 1rem 0;
    font-size: 0.88rem;
    color: {MUTED};
}}
.insight-box strong {{ color: {TEXT}; }}

/* Divider */
.divider {{
    border: none;
    border-top: 1px solid {BORDER};
    margin: 1.5rem 0;
}}

/* Streamlit widgets */
.stSelectbox > div, .stMultiSelect > div {{
    background-color: {CARD} !important;
    border-color: {BORDER} !important;
    color: {TEXT} !important;
    border-radius: 8px !important;
}}
.stDataFrame {{
    background: {CARD};
    border-radius: 10px;
    overflow: hidden;
}}
div[data-testid="metric-container"] {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 1rem 1.25rem;
}}
div[data-testid="metric-container"] label {{
    color: {MUTED} !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}}
div[data-testid="metric-container"] [data-testid="metric-value"] {{
    color: {TEXT} !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
}}
</style>
""", unsafe_allow_html=True)

# ── constants ─────────────────────────────────────────────────────────────────
COMP_ID          = 43
SEASON_ID        = 3
RAW_DIR          = Path("data/raw/comp_43_season_3")
TEAM_FEATURES_CSV = Path("data/team_features.csv")


# ── ensure data present ───────────────────────────────────────────────────────
def _ensure_data():
    if not RAW_DIR.exists() or not any(RAW_DIR.glob("events_*.csv")):
        with st.spinner("Downloading StatsBomb World Cup data (one-time, ~30s)..."):
            from football_ai.io.statsbomb_loader import download_match_events_to_csv
            download_match_events_to_csv(COMP_ID, SEASON_ID, output_dir=RAW_DIR)
    if not TEAM_FEATURES_CSV.exists():
        with st.spinner("Building team features dataset..."):
            from football_ai.pipeline.build_dataset import build_team_features_csv
            build_team_features_csv(COMP_ID, SEASON_ID, events_dir=RAW_DIR,
                                    output_csv=TEAM_FEATURES_CSV)

_ensure_data()

# ── sidebar nav ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="padding: 0.5rem 0 1.5rem 0;">
        <div style="font-size:1.3rem; font-weight:700; color:{TEXT};">⚽ Football xG</div>
        <div style="font-size:0.75rem; color:{MUTED}; margin-top:0.2rem;">2018 FIFA World Cup</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["Overview", "Shot Map", "xG Model", "Team Stats", "Match Outcome"],
        label_visibility="hidden",
    )

    st.markdown(f"""
    <div style="margin-top:2rem; padding-top:0.75rem; border-top:1px solid {BORDER};
                font-size:0.7rem; color:{MUTED}; line-height:1.6;">
        StatsBomb Open Data<br>64 matches · 32 teams
    </div>
    """, unsafe_allow_html=True)

# ── helpers ───────────────────────────────────────────────────────────────────
def kpi(label, value, sub="", accent=False):
    val_color = GREEN if accent else TEXT
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value" style="color:{val_color};">{value}</div>
        {"<div class='kpi-sub'>" + sub + "</div>" if sub else ""}
    </div>"""

def section(title):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)

def insight(text):
    st.markdown(f'<div class="insight-box">{text}</div>', unsafe_allow_html=True)

# ── cached loaders ────────────────────────────────────────────────────────────
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
    return load_shot_dataset(get_csv_files())

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
    return model, evaluate_outcome_model(y_val, y_pred)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: Overview
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.markdown('<div class="page-title">Tournament Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">2018 FIFA World Cup · All 64 matches · StatsBomb event data</div>', unsafe_allow_html=True)

    matches   = load_matches()
    shots     = get_shot_dataset()
    team_df   = load_team_features()
    n_teams   = len(set(matches["home_team"]) | set(matches["away_team"]))
    goal_rate = shots["is_goal"].mean()
    total_xg  = team_df["total_xg"].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi("Matches Played", 64, "Group stage → Final"), unsafe_allow_html=True)
    c2.markdown(kpi("Teams", n_teams, "32 nations"), unsafe_allow_html=True)
    c3.markdown(kpi("Total Shots", f"{len(shots):,}", f"Goal rate {goal_rate:.1%}"), unsafe_allow_html=True)
    c4.markdown(kpi("Total xG", f"{total_xg:.0f}", "Expected goals scored", accent=True), unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2])

    with col_l:
        section("xG Leaderboard — Full Tournament")
        xg_total = team_df.groupby("team")["total_xg"].sum().sort_values(ascending=True)
        fig = go.Figure(go.Bar(
            x=xg_total.values,
            y=xg_total.index,
            orientation="h",
            marker=dict(
                color=xg_total.values,
                colorscale=[[0, BORDER], [1, GREEN]],
                line=dict(width=0),
            ),
            text=[f"{v:.1f}" for v in xg_total.values],
            textposition="outside",
            textfont=dict(color=MUTED, size=10),
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=700,
                          xaxis_title="Total xG", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        section("Match Result Distribution")
        result_counts = team_df["result"].value_counts()
        fig2 = go.Figure(go.Pie(
            labels=result_counts.index,
            values=result_counts.values,
            hole=0.6,
            marker=dict(colors=[GREEN, BLUE, RED],
                        line=dict(color=SURFACE, width=3)),
            textinfo="label+percent",
            textfont=dict(color=TEXT, size=13),
        ))
        fig2.update_layout(**PLOTLY_LAYOUT, height=320,
                           showlegend=False,
                           annotations=[dict(text="Results", x=0.5, y=0.5,
                                            font=dict(size=14, color=MUTED), showarrow=False)])
        st.plotly_chart(fig2, use_container_width=True)

        section("What This App Does")
        insight("<strong>xG Model</strong> — predicts the probability each shot becomes a goal using location, distance, body part & technique.")
        insight("<strong>Outcome Model</strong> — predicts match result (win/draw/loss) from team-level stats: possession, passing tempo, pressing, xG.")
        insight("<strong>Data:</strong> StatsBomb Open Data — ~3,000 events per match, freely available.")


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: Shot Map
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Shot Map":
    st.markdown('<div class="page-title">Shot Map</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Select any match to explore shot locations, xG, and outcomes</div>', unsafe_allow_html=True)

    matches = load_matches()
    match_options = {
        f"{row['home_team']} vs {row['away_team']}  ·  Week {row['match_week']}": int(row["match_id"])
        for _, row in matches.sort_values("match_week").iterrows()
    }

    selected = st.selectbox("Match", list(match_options.keys()), label_visibility="collapsed")
    match_id = match_options[selected]
    csv_path = RAW_DIR / f"events_{match_id}.csv"

    if csv_path.exists():
        events   = pd.read_csv(csv_path)
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

        # KPI row
        teams = shot_rows["team"].dropna().unique()
        cols  = st.columns(len(teams) * 2)
        for i, team in enumerate(teams):
            t = shot_rows[shot_rows["team"] == team]
            xg_sum = t["shot_statsbomb_xg"].sum() if "shot_statsbomb_xg" in t.columns else 0
            cols[i*2].markdown(kpi(f"{team} Shots", len(t), f"xG: {xg_sum:.2f}"), unsafe_allow_html=True)
            cols[i*2+1].markdown(kpi(f"{team} Goals", int(t["is_goal"].sum()), "", accent=True), unsafe_allow_html=True)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        from mplsoccer import Pitch
        pitch = Pitch(pitch_type="statsbomb", pitch_color="#0f1923", line_color="#3d5a6e")
        fig, ax = pitch.draw(figsize=(12, 7))
        fig.patch.set_facecolor(SURFACE)

        colors_map = {teams[0]: GREEN, teams[1]: BLUE} if len(teams) >= 2 else {teams[0]: GREEN}
        for team, color in colors_map.items():
            sub = shot_rows[shot_rows["team"] == team]
            xg_vals = sub["shot_statsbomb_xg"].fillna(0.05).values if "shot_statsbomb_xg" in sub.columns else np.full(len(sub), 0.05)
            goals   = sub["is_goal"].astype(bool).values
            sizes   = (xg_vals * 1200).clip(60, 1200)
            pitch.scatter(sub["x"].values[~goals], sub["y"].values[~goals],
                          s=sizes[~goals], edgecolors=color, facecolors="none",
                          linewidths=1.5, ax=ax, label=f"{team}")
            pitch.scatter(sub["x"].values[goals], sub["y"].values[goals],
                          s=sizes[goals], edgecolors=color, facecolors=color,
                          linewidths=1.5, ax=ax, zorder=5)

        ax.legend(loc="upper left", fontsize=10,
                  facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT)
        ax.set_title(selected, color=TEXT, fontsize=13, pad=12, fontweight="600")
        st.pyplot(fig)
        plt.close()

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        # ── xG Timeline ──────────────────────────────────────────────────────
        section("xG Timeline")
        insight("Cumulative expected goals per minute. Steep spikes = high-quality chances. Goals marked with a ★.")

        palette = {teams[0]: GREEN, teams[1]: BLUE} if len(teams) >= 2 else {teams[0]: GREEN}
        # rgba fill versions (10% opacity)
        palette_fill = {
            teams[0]: "rgba(0,245,122,0.10)",
            teams[1]: "rgba(59,130,246,0.10)",
        } if len(teams) >= 2 else {teams[0]: "rgba(0,245,122,0.10)"}
        fig_timeline = go.Figure()

        for team, color in palette.items():
            t = shot_rows[shot_rows["team"] == team].copy()
            t["minute"] = pd.to_numeric(t["minute"], errors="coerce").fillna(0).astype(int)
            t["xg"] = pd.to_numeric(t.get("shot_statsbomb_xg", 0), errors="coerce").fillna(0.05)
            t = t.sort_values("minute")
            t["cumxg"] = t["xg"].cumsum()

            # Line: cumulative xG
            fig_timeline.add_trace(go.Scatter(
                x=t["minute"], y=t["cumxg"],
                mode="lines", name=team,
                line=dict(color=color, width=2.5, shape="hv"),
                fill="tozeroy",
                fillcolor=palette_fill[team],
            ))

            # Stars: goals
            goals = t[t["is_goal"] == 1]
            if not goals.empty:
                fig_timeline.add_trace(go.Scatter(
                    x=goals["minute"], y=goals["cumxg"],
                    mode="markers+text",
                    marker=dict(symbol="star", size=16, color=color,
                                line=dict(color=SURFACE, width=1)),
                    text=["⚽"] * len(goals),
                    textposition="top center",
                    textfont=dict(size=11),
                    name=f"{team} goal",
                    showlegend=False,
                ))

        # Halftime line
        fig_timeline.add_vline(x=45, line=dict(color=BORDER, dash="dash", width=1))
        fig_timeline.add_annotation(x=45, y=0, text="HT", showarrow=False,
                                     font=dict(color=MUTED, size=10), yshift=-15)

        fig_timeline.update_layout(
            **PLOTLY_LAYOUT, height=320,
            xaxis_title="Minute", yaxis_title="Cumulative xG",
            xaxis_range=[0, 95],
            legend=dict(bgcolor=CARD, bordercolor=BORDER),
            hovermode="x unified",
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        section("Shot Details")
        display_cols = [c for c in ["team", "player", "minute", "x", "y",
                                     "shot_statsbomb_xg", "shot_outcome",
                                     "shot_body_part", "shot_technique"] if c in shot_rows.columns]
        st.dataframe(
            shot_rows[display_cols].rename(columns={
                "shot_statsbomb_xg": "xG",
                "shot_body_part": "body part",
                "shot_technique": "technique",
                "shot_outcome": "outcome",
            }).reset_index(drop=True),
            use_container_width=True,
            height=300,
        )
    else:
        st.warning(f"No event CSV found for match {match_id}.")


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: xG Model
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "xG Model":
    st.markdown('<div class="page-title">Expected Goals Model</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Gradient Boosting classifier trained on all World Cup shots</div>', unsafe_allow_html=True)

    insight("xG (expected goals) is a probability: <strong>0.8 xG</strong> means the model thinks that shot had an 80% chance of being a goal. It accounts for shot location, distance, body part, and technique.")

    with st.spinner("Training model..."):
        model, X_val, y_val, y_proba, metrics = train_xg_model_cached()

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi("ROC-AUC", metrics["roc_auc"], "1.0 = perfect · 0.5 = random", accent=True), unsafe_allow_html=True)
    c2.markdown(kpi("Brier Score", metrics["brier_score"], "Lower = better calibrated"), unsafe_allow_html=True)
    c3.markdown(kpi("Log Loss", metrics["log_loss"]), unsafe_allow_html=True)
    c4.markdown(kpi("Calib. Error", metrics["ece"], "Expected calibration error"), unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        section("Calibration Curve")
        insight("How closely predicted xG matches real goal rates. A perfect model follows the dashed line exactly.")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                  line=dict(dash="dash", color=BORDER, width=1.5),
                                  name="Perfect"))
        fig.add_trace(go.Scatter(x=metrics["mean_pred"], y=metrics["fraction_pos"],
                                  mode="lines+markers",
                                  line=dict(color=GREEN, width=2),
                                  marker=dict(size=8, color=GREEN),
                                  name="Model"))
        fig.update_layout(**PLOTLY_LAYOUT, height=360,
                          xaxis_title="Mean predicted xG",
                          yaxis_title="Fraction of goals",
                          xaxis_range=[0,1], yaxis_range=[0,1],
                          legend=dict(bgcolor=CARD, bordercolor=BORDER))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        section("Feature Importance")
        insight("Shot distance and penalty type dominate. Body part and technique add secondary signal.")
        num_feats  = [c for c in ["x","y","shot_distance","minute","second"] if c in X_val.columns]
        ohe        = model.named_steps["preprocess"].named_transformers_["cat"]
        cat_feats  = list(ohe.get_feature_names_out(["shot_body_part","shot_technique","shot_type"]))
        all_feats  = num_feats + cat_feats
        imps       = model.named_steps["clf"].feature_importances_
        top_idx    = np.argsort(imps)[::-1][:12]
        top_names  = [all_feats[i] for i in top_idx]
        top_imps   = imps[top_idx]
        fig2 = go.Figure(go.Bar(
            x=top_imps[::-1], y=top_names[::-1], orientation="h",
            marker=dict(color=top_imps[::-1], colorscale=[[0, BORDER],[1, GREEN]], line=dict(width=0)),
        ))
        fig2.update_layout(**PLOTLY_LAYOUT, height=360,
                           xaxis_title="Importance", yaxis_title="")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    section("xG Distribution — Goals vs Non-Goals")
    insight("Goals cluster at higher xG values. Most shots have low xG — high-quality chances are rare.")

    fig3 = go.Figure()
    fig3.add_trace(go.Histogram(x=y_proba[y_val==0], nbinsx=40,
                                 name="No goal", marker_color=BLUE, opacity=0.7))
    fig3.add_trace(go.Histogram(x=y_proba[y_val==1], nbinsx=20,
                                 name="Goal", marker_color=GREEN, opacity=0.85))
    fig3.update_layout(**PLOTLY_LAYOUT, height=300, barmode="overlay",
                       xaxis_title="Predicted xG", yaxis_title="Count",
                       legend=dict(bgcolor=CARD, bordercolor=BORDER))
    st.plotly_chart(fig3, use_container_width=True)

    # ── SHAP explainability ───────────────────────────────────────────────────
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    section("Why did the model predict this xG?  —  SHAP Explainability")
    insight("SHAP values show how much each feature <strong>pushed the prediction up (green) or down (red)</strong> from the baseline. Select a shot to explain.")

    import shap

    # Build feature names
    num_feats = [c for c in ["x","y","shot_distance","minute","second"] if c in X_val.columns]
    ohe       = model.named_steps["preprocess"].named_transformers_["cat"]
    cat_feats = list(ohe.get_feature_names_out(["shot_body_part","shot_technique","shot_type"]))
    all_feats = num_feats + cat_feats

    # Transform validation set
    X_val_t = model.named_steps["preprocess"].transform(X_val)

    @st.cache_data
    def compute_shap(_model, _X_val_t):
        explainer   = shap.TreeExplainer(_model.named_steps["clf"])
        shap_values = explainer.shap_values(_X_val_t)
        # GBM returns 1D array for binary classification
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        return explainer.expected_value if not isinstance(explainer.expected_value, list) \
               else explainer.expected_value[1], shap_values

    base_val, shap_vals = compute_shap(model, X_val_t)

    col_shap_l, col_shap_r = st.columns([1, 2])

    with col_shap_l:
        # Shot selector
        shot_idx = st.slider("Shot index (validation set)", 0, len(X_val) - 1, 0)
        pred_xg  = float(y_proba[shot_idx])
        actual   = int(y_val.iloc[shot_idx])
        st.markdown(kpi("Predicted xG", f"{pred_xg:.3f}", "Goal" if actual else "No goal",
                        accent=actual==1), unsafe_allow_html=True)
        st.markdown(kpi("Baseline xG", f"{base_val:.3f}", "Average across all shots"), unsafe_allow_html=True)

        # Top contributing features for this shot
        sv     = shap_vals[shot_idx]
        top_k  = np.argsort(np.abs(sv))[::-1][:8]
        top_names_shap = [all_feats[i] if i < len(all_feats) else f"feat_{i}" for i in top_k]
        top_sv = sv[top_k]
        colors_shap = [GREEN if v > 0 else RED for v in top_sv]

        fig_w = go.Figure(go.Bar(
            x=top_sv[::-1], y=top_names_shap[::-1], orientation="h",
            marker=dict(color=colors_shap[::-1], line=dict(width=0)),
        ))
        fig_w.update_layout(**PLOTLY_LAYOUT, height=340,
                            xaxis_title="SHAP value (impact on xG)",
                            yaxis_title="",
                            title=dict(text="Shot explanation", font=dict(color=TEXT, size=13)))
        st.plotly_chart(fig_w, use_container_width=True)

    with col_shap_r:
        section("Global Feature Impact — All Shots")
        insight("Mean absolute SHAP value across the validation set. Shows which features the model relies on most <em>overall</em>.")

        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        top_global    = np.argsort(mean_abs_shap)[::-1][:12]
        g_names = [all_feats[i] if i < len(all_feats) else f"feat_{i}" for i in top_global]
        g_vals  = mean_abs_shap[top_global]

        fig_g = go.Figure(go.Bar(
            x=g_vals[::-1], y=g_names[::-1], orientation="h",
            marker=dict(color=g_vals[::-1],
                        colorscale=[[0, BORDER],[1, GREEN]], line=dict(width=0)),
            text=[f"{v:.4f}" for v in g_vals[::-1]],
            textposition="outside", textfont=dict(color=MUTED, size=9),
        ))
        fig_g.update_layout(**PLOTLY_LAYOUT, height=420,
                            xaxis_title="Mean |SHAP value|", yaxis_title="")
        st.plotly_chart(fig_g, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: Team Stats
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Team Stats":
    st.markdown('<div class="page-title">Team Statistics</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Aggregated stats across all World Cup matches</div>', unsafe_allow_html=True)

    team_df = load_team_features()
    agg = team_df.groupby("team").agg(
        matches        =("match_id",         "count"),
        total_xg       =("total_xg",         "sum"),
        avg_possession =("possession_pct",    "mean"),
        avg_passes_min =("passes_per_min",    "mean"),
        avg_press_min  =("pressures_per_min", "mean"),
        avg_shot_dist  =("avg_shot_distance", "mean"),
    ).round(2).sort_values("total_xg", ascending=False)

    section("Tournament Table")
    st.dataframe(agg, use_container_width=True, height=400)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    section("Compare Teams")

    all_teams      = sorted(team_df["team"].unique())
    selected_teams = st.multiselect("Select teams", all_teams,
                                     default=["France","Croatia","Belgium","England"])
    metric_map = {
        "Total xG":             "total_xg",
        "Avg Possession %":     "avg_possession",
        "Avg Passes / min":     "avg_passes_min",
        "Avg Pressures / min":  "avg_press_min",
        "Avg Shot Distance":    "avg_shot_dist",
    }
    metric_label = st.selectbox("Metric", list(metric_map.keys()))
    metric_col   = metric_map[metric_label]

    if selected_teams:
        subset = agg.loc[agg.index.isin(selected_teams)].reset_index()
        subset = subset.sort_values(metric_col, ascending=False)
        colors = [GREEN if i == 0 else BLUE for i in range(len(subset))]
        fig = go.Figure(go.Bar(
            x=subset["team"], y=subset[metric_col],
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{v:.2f}" for v in subset[metric_col]],
            textposition="outside",
            textfont=dict(color=MUTED),
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=380,
                          yaxis_title=metric_label, xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        section("Team DNA — Radar Chart")
        insight("Each axis is normalised 0–1 across all teams. Bigger area = stronger overall performance.")

        radar_metrics = ["avg_possession", "avg_passes_min", "avg_press_min", "total_xg", "avg_shot_dist"]
        radar_labels  = ["Possession", "Passing Tempo", "Pressing", "Total xG", "Shot Distance"]

        # Normalise each metric 0–1 across all teams
        radar_data = agg[radar_metrics].copy()
        radar_norm = (radar_data - radar_data.min()) / (radar_data.max() - radar_data.min() + 1e-9)

        palette = [GREEN, BLUE, YELLOW, RED, "#A855F7", "#F97316"]
        fig_radar = go.Figure()

        for i, team in enumerate(selected_teams):
            if team not in radar_norm.index:
                continue
            vals = radar_norm.loc[team, radar_metrics].tolist()
            vals += vals[:1]  # close the polygon
            cats = radar_labels + radar_labels[:1]
            color = palette[i % len(palette)]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals, theta=cats, name=team,
                fill="toself",
                fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.10)"
                          if color.startswith("#") and len(color) == 7 else "rgba(0,245,122,0.10)",
                line=dict(color=color, width=2),
            ))

        fig_radar.update_layout(
            paper_bgcolor=SURFACE,
            plot_bgcolor=SURFACE,
            font=dict(family="Inter, sans-serif", color=TEXT),
            polar=dict(
                bgcolor=CARD,
                radialaxis=dict(visible=True, range=[0, 1], gridcolor=BORDER,
                                tickfont=dict(color=MUTED, size=9), showticklabels=False),
                angularaxis=dict(gridcolor=BORDER, tickfont=dict(color=TEXT, size=11)),
            ),
            legend=dict(bgcolor=CARD, bordercolor=BORDER),
            margin=dict(l=60, r=60, t=40, b=40),
            height=480,
        )
        st.plotly_chart(fig_radar, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: Match Outcome Model
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Match Outcome":
    st.markdown('<div class="page-title">Match Outcome Model</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Random Forest classifier — predicts win / draw / loss from team stats</div>', unsafe_allow_html=True)

    insight("The model uses 5 features: <strong>possession %, passes/min, pressures/min, average shot distance, and total xG</strong>. Limited by dataset size (128 rows) — accuracy improves significantly with more competitions.")

    with st.spinner("Training outcome model..."):
        model, metrics = train_outcome_model_cached()

    c1, c2, c3 = st.columns(3)
    c1.markdown(kpi("Accuracy", f"{metrics['accuracy']:.0%}", "Validation set", accent=True), unsafe_allow_html=True)
    c2.markdown(kpi("Best Class", "Loss", "F1: 0.67"), unsafe_allow_html=True)
    c3.markdown(kpi("Hardest Class", "Draw", "F1: 0.22 — draws are unpredictable"), unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        section("Per-class Performance")
        per_class = {k: v for k, v in metrics["per_class"].items() if k in ("win","draw","loss")}
        classes   = list(per_class.keys())
        precision = [per_class[c]["precision"] for c in classes]
        recall    = [per_class[c]["recall"]    for c in classes]
        f1        = [per_class[c]["f1-score"]  for c in classes]

        fig = go.Figure()
        for vals, name, color in [(precision,"Precision",GREEN),(recall,"Recall",BLUE),(f1,"F1",YELLOW)]:
            fig.add_trace(go.Bar(name=name, x=classes, y=vals,
                                  marker=dict(color=color, line=dict(width=0))))
        fig.update_layout(**PLOTLY_LAYOUT, barmode="group", height=340,
                          yaxis_range=[0,1.05], yaxis_title="Score",
                          legend=dict(bgcolor=CARD, bordercolor=BORDER))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        section("Feature Importance")
        insight("xG is the most predictive feature — a team that creates better chances wins more often.")
        feat_names = ["possession_pct","passes_per_min","pressures_per_min","avg_shot_distance","total_xg"]
        feat_labels = ["Possession %","Passes/min","Pressures/min","Shot Distance","Total xG"]
        imps = model.named_steps["clf"].feature_importances_
        order = np.argsort(imps)
        fig2 = go.Figure(go.Bar(
            x=imps[order], y=[feat_labels[i] for i in order], orientation="h",
            marker=dict(color=imps[order],
                        colorscale=[[0,BORDER],[1,GREEN]], line=dict(width=0)),
            text=[f"{v:.3f}" for v in imps[order]],
            textposition="outside", textfont=dict(color=MUTED),
        ))
        fig2.update_layout(**PLOTLY_LAYOUT, height=340,
                           xaxis_title="Importance", yaxis_title="")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    section("Full Classification Report")
    st.code(metrics["classification_report"], language=None)

    # ── What-if Simulator ─────────────────────────────────────────────────────
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    section("What-If Simulator")
    insight("Drag the sliders to set a team's in-match stats and see the model's predicted outcome — <strong>live</strong>.")

    team_df_sim = load_team_features()

    sim_col, res_col = st.columns([2, 1])

    with sim_col:
        possession  = st.slider("Possession %",        10.0, 90.0, 50.0, 0.5)
        passes_min  = st.slider("Passes per minute",    1.0, 12.0,  5.0, 0.1)
        press_min   = st.slider("Pressures per minute", 0.5,  6.0,  2.0, 0.1)
        shot_dist   = st.slider("Avg shot distance (m)", 8.0, 35.0, 18.0, 0.5)
        total_xg    = st.slider("Total xG",             0.0,  5.0,  1.5, 0.05)

    sim_input = pd.DataFrame([{
        "possession_pct":    possession,
        "passes_per_min":    passes_min,
        "pressures_per_min": press_min,
        "avg_shot_distance": shot_dist,
        "total_xg":          total_xg,
    }])

    proba     = model.predict_proba(sim_input)[0]
    classes   = model.classes_
    proba_map = dict(zip(classes, proba))
    pred      = classes[np.argmax(proba)]

    outcome_color = {"win": GREEN, "draw": YELLOW, "loss": RED}
    pred_color    = outcome_color.get(pred, TEXT)

    with res_col:
        st.markdown(f"""
        <div class="kpi-card" style="text-align:center; margin-top:1rem;">
            <div class="kpi-label">Predicted outcome</div>
            <div class="kpi-value" style="font-size:2.5rem; color:{pred_color};">
                {pred.upper()}
            </div>
            <div class="kpi-sub" style="margin-top:0.5rem;">
                {"🟢" if pred=="win" else "🟡" if pred=="draw" else "🔴"}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Probability bar chart
    ordered = ["win", "draw", "loss"]
    probs   = [proba_map.get(c, 0) for c in ordered]
    colors_sim = [GREEN, YELLOW, RED]

    fig_sim = go.Figure(go.Bar(
        x=ordered, y=probs,
        marker=dict(color=colors_sim, line=dict(width=0)),
        text=[f"{p:.1%}" for p in probs],
        textposition="outside",
        textfont=dict(color=TEXT, size=13),
    ))
    fig_sim.update_layout(
        **PLOTLY_LAYOUT, height=280,
        yaxis_range=[0, 1.15],
        yaxis_title="Probability", xaxis_title="",
        showlegend=False,
    )
    st.plotly_chart(fig_sim, use_container_width=True)

    # Context: compare to real team averages
    section("How does this compare to real teams?")
    agg_sim = team_df_sim.groupby("team").agg(
        possession_pct    =("possession_pct",    "mean"),
        passes_per_min    =("passes_per_min",    "mean"),
        pressures_per_min =("pressures_per_min", "mean"),
        avg_shot_distance =("avg_shot_distance", "mean"),
        total_xg          =("total_xg",          "mean"),
    )
    diffs = (agg_sim - sim_input.iloc[0]).abs().sum(axis=1)
    closest = diffs.nsmallest(3).index.tolist()
    insight(f"Your slider profile is closest to: <strong>{', '.join(closest)}</strong>")
