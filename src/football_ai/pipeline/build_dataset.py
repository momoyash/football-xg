from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import pandas as pd
from statsbombpy import sb
from tqdm import tqdm

from football_ai.preprocessing.feature_engineering import compute_team_features


def _get_match_results(competition_id: int, season_id: int) -> dict[int, dict[str, str]]:
    """
    Return a mapping of match_id -> {team_name: result} for all matches in a
    competition/season. Result values are 'win', 'draw', or 'loss'.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        matches = sb.matches(competition_id=competition_id, season_id=season_id)

    results: dict[int, dict[str, str]] = {}
    for _, row in matches.iterrows():
        mid = int(row["match_id"])
        home = str(row["home_team"])
        away = str(row["away_team"])
        hs = int(row["home_score"])
        as_ = int(row["away_score"])

        if hs > as_:
            results[mid] = {home: "win", away: "loss"}
        elif hs < as_:
            results[mid] = {home: "loss", away: "win"}
        else:
            results[mid] = {home: "draw", away: "draw"}

    return results


def build_team_features_csv(
    competition_id: int = 43,
    season_id: int = 3,
    events_dir: str | Path | None = None,
    output_csv: str | Path = "data/team_features.csv",
) -> Path:
    """
    Iterate over all match event CSVs for a competition/season, compute
    team-level features per match, attach match result labels, and write
    a single consolidated CSV.

    Parameters
    ----------
    competition_id : int
        StatsBomb competition ID (default 43 = 2018 World Cup).
    season_id : int
        StatsBomb season ID (default 3).
    events_dir : str | Path, optional
        Directory containing events_*.csv files. Defaults to
        data/raw/comp_{competition_id}_season_{season_id}/.
    output_csv : str | Path
        Destination CSV path.
    """
    if events_dir is None:
        events_dir = Path("data") / "raw" / f"comp_{competition_id}_season_{season_id}"
    events_dir = Path(events_dir)

    csv_files = sorted(events_dir.glob("events_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No event CSVs found in {events_dir}")

    print(f"Found {len(csv_files)} match CSVs. Fetching match results...")
    match_results = _get_match_results(competition_id, season_id)

    rows: list[pd.DataFrame] = []
    for csv_path in tqdm(csv_files, desc="Building features"):
        match_id = int(csv_path.stem.split("_")[1])
        events = pd.read_csv(csv_path)

        features = compute_team_features(events)
        features = features.reset_index().rename(columns={"index": "team"})
        features["match_id"] = match_id

        if match_id in match_results:
            features["result"] = features["team"].map(match_results[match_id])
        else:
            features["result"] = None

        rows.append(features)

    combined = pd.concat(rows, ignore_index=True)
    combined = combined.dropna(subset=["result"])

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_csv, index=False)

    n_matches = combined["match_id"].nunique()
    print(f"Saved {len(combined)} rows ({n_matches} matches) to {output_csv.resolve()}")
    return output_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build team-level features CSV from StatsBomb event CSVs."
    )
    parser.add_argument("--competition-id", type=int, default=43)
    parser.add_argument("--season-id", type=int, default=3)
    parser.add_argument("--events-dir", type=str, default=None)
    parser.add_argument(
        "--output-csv", type=str, default="data/team_features.csv"
    )
    args = parser.parse_args()

    build_team_features_csv(
        competition_id=args.competition_id,
        season_id=args.season_id,
        events_dir=args.events_dir,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()
