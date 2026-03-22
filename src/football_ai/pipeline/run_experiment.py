from __future__ import annotations

import argparse
from pathlib import Path

from football_ai.io.statsbomb_loader import download_match_events_to_csv
from football_ai.modeling.xg_model import train_xg_pipeline_from_events
from football_ai.modeling.train import train_match_outcome_model
from football_ai.pipeline.build_dataset import build_team_features_csv


def run_experiment(
    competition_id: int = 43,
    season_id: int = 3,
    download: bool = False,
    team_features_csv: str | Path = "data/team_features.csv",
    xg_model_out: str | Path = "models/artifacts/xg_model.joblib",
    outcome_model_out: str | Path = "models/artifacts/match_outcome_model.joblib",
) -> None:
    """
    End-to-end pipeline:
      1. (Optionally) download StatsBomb event CSVs
      2. Build team-level features CSV with match result labels
      3. Train xG model
      4. Train match outcome model
    """
    # Step 1 — download raw data
    if download:
        print(f"\n[1/4] Downloading events for competition={competition_id} season={season_id}...")
        events_dir = download_match_events_to_csv(competition_id, season_id)
    else:
        events_dir = Path("data") / "raw" / f"comp_{competition_id}_season_{season_id}"
        print(f"\n[1/4] Skipping download, using existing data in {events_dir}")

    # Step 2 — build team features
    print("\n[2/4] Building team features dataset...")
    build_team_features_csv(
        competition_id=competition_id,
        season_id=season_id,
        events_dir=events_dir,
        output_csv=team_features_csv,
    )

    # Step 3 — train xG model
    events_glob = str(events_dir / "events_*.csv")
    print(f"\n[3/4] Training xG model (glob: {events_glob})...")
    train_xg_pipeline_from_events(
        events_glob=events_glob,
        model_out=xg_model_out,
    )

    # Step 4 — train match outcome model
    print("\n[4/4] Training match outcome model...")
    train_match_outcome_model(
        features_csv=team_features_csv,
        model_out=outcome_model_out,
    )

    print("\nDone.")
    print(f"  xG model       -> {Path(xg_model_out).resolve()}")
    print(f"  Outcome model  -> {Path(outcome_model_out).resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full football analytics pipeline."
    )
    parser.add_argument("--competition-id", type=int, default=43)
    parser.add_argument("--season-id", type=int, default=3)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download event CSVs from StatsBomb before running (skipped by default).",
    )
    parser.add_argument(
        "--team-features-csv",
        type=str,
        default="data/team_features.csv",
    )
    parser.add_argument(
        "--xg-model-out",
        type=str,
        default="models/artifacts/xg_model.joblib",
    )
    parser.add_argument(
        "--outcome-model-out",
        type=str,
        default="models/artifacts/match_outcome_model.joblib",
    )
    args = parser.parse_args()

    run_experiment(
        competition_id=args.competition_id,
        season_id=args.season_id,
        download=args.download,
        team_features_csv=args.team_features_csv,
        xg_model_out=args.xg_model_out,
        outcome_model_out=args.outcome_model_out,
    )


if __name__ == "__main__":
    main()
