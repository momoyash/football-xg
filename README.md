# Football Analytics AI

A Python project for football analytics and machine learning on **StatsBomb** event data: data ingestion, exploration, team- and shot-level features, expected goals (xG) modeling, match-outcome prediction, and tactical-state classification (with labels).

---

## Quick start

```bash
cd football-analytics-ai
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

**Run from the project root** (`football-analytics-ai`) so paths like `data/raw/...` resolve correctly.

---

## Project layout

| Path | Purpose |
|------|---------|
| `data/raw/` | Downloaded StatsBomb event CSVs (e.g. per match) |
| `data/interim/` | Intermediate cleaned or labeled data |
| `data/processed/` | Feature tables ready for modeling |
| `models/artifacts/` | Saved models (`.joblib`) |
| `notebooks/` | Jupyter workflows (exploration, features, experiments) |
| `src/football_ai/` | Importable package (`io`, `preprocessing`, `modeling`, `evaluation`) |
| `docs/` | Project history, roadmap, and detailed guides |

---

## What’s implemented

| Area | Description |
|------|-------------|
| **Data download** | `statsbombpy` → CSV per match (`io/statsbomb_loader.py`) |
| **Exploration** | `notebooks/01_exploration_statsbomb.ipynb` |
| **Aggregate stats** | `notebooks/02_feature_prototyping.ipynb` — load many event CSVs, match-level counts |
| **Match summary CLI** | `evaluation/match_summary.py` — passes, shots, pressures, xG, possession from one CSV |
| **Team features** | `preprocessing/feature_engineering.py` — possession %, rates per minute, shot distance, total xG |
| **Shot dataset** | `build_shot_dataset()` — rows for xG-style modeling |
| **xG model** | `modeling/xg_model.py` — train classifier on shots → goal; outputs probability as xG |
| **Match outcome** | `modeling/train.py` — sklearn pipeline on a **team-features CSV** + `result` column |
| **Tactical state** | `modeling/tactical_state.py` — sequence windows → classifier (**requires** `tactical_state` labels on events) |

Full task log and future plans: **[docs/PROJECT_HISTORY_AND_ROADMAP.md](docs/PROJECT_HISTORY_AND_ROADMAP.md)**.

---

## Common commands

**Download events (Open Data)**

```bash
python -m src.football_ai.io.statsbomb_loader <competition_id> <season_id>
```

**Summarize one match CSV**

```bash
python -m src.football_ai.evaluation.match_summary data/raw/comp_43_season_3/events_7525.csv
```

**Train xG model (PowerShell — use backtick for line continuation)**

```powershell
python -m src.football_ai.modeling.xg_model `
  --events-glob "data/raw/comp_43_season_3/events_*.csv" `
  --model-out models/artifacts/xg_model.joblib
```

**Train match-outcome model** (needs `data/team_features.csv` or your path with columns `result` + numeric features)

```powershell
python -m src.football_ai.modeling.train path/to/team_features.csv --target-column result
```

**Train tactical-state model** (event CSVs must include a `tactical_state` column)

```powershell
python -m src.football_ai.modeling.tactical_state `
  --events-glob "data/raw/**/*.csv" `
  --state-col tactical_state
```

---

## Using the package in notebooks

If `ModuleNotFoundError: football_ai` appears, either:

1. Install in editable mode (recommended once `pyproject.toml` is configured), or  
2. Add `src` to the path:

```python
import sys
from pathlib import Path
sys.path.append(str(Path(r"E:\path\to\football-analytics-ai") / "src"))
```

**Windows paths in strings:** use `r"E:\..."`, double backslashes, or forward slashes — otherwise `\f`, `\t`, etc. break the path.

---

## Dependencies

See `requirements.txt` — includes **pandas**, **numpy**, **scikit-learn**, **statsbombpy**, **mplsoccer**, **streamlit**, **PyTorch**, and notebook/plotting helpers.

---

## License & data

StatsBomb Open Data is subject to [StatsBomb’s terms](https://github.com/statsbomb/open-data). This repo is a personal/analytics scaffold; add your own license if you distribute code.
