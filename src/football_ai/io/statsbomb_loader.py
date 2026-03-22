from __future__ import annotations

from pathlib import Path
from typing import Optional

from statsbombpy import sb


def download_match_events_to_csv(
    competition_id: int,
    season_id: int,
    output_dir: Optional[str | Path] = None,
    overwrite: bool = False,
) -> Path:
    """
    Download all match events for a given competition and season from StatsBomb
    using statsbombpy and save each match's events as a CSV file.

    Parameters
    ----------
    competition_id : int
        StatsBomb competition ID (e.g. 43 for WSL, 11 for La Liga in Open Data).
    season_id : int
        StatsBomb season ID within the competition.
    output_dir : str | Path, optional
        Directory where CSV files will be written. If None, defaults to
        "data/raw/comp_{competition_id}_season_{season_id}" relative to cwd.
    overwrite : bool, default False
        If False, existing CSV files will be skipped.

    Returns
    -------
    Path
        The directory containing the downloaded CSV files.
    """
    if output_dir is None:
        output_dir = Path("data") / "raw" / f"comp_{competition_id}_season_{season_id}"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Get list of matches for this competition + season
    matches = sb.matches(competition_id=competition_id, season_id=season_id)

    for _, match in matches.iterrows():
        match_id = int(match["match_id"])
        out_path = output_dir / f"events_{match_id}.csv"

        if out_path.exists() and not overwrite:
            continue

        events = sb.events(match_id=match_id)
        events.to_csv(out_path, index=False)

    return output_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download StatsBomb match events and save as CSV files."
    )
    parser.add_argument("competition_id", type=int, help="StatsBomb competition ID")
    parser.add_argument("season_id", type=int, help="StatsBomb season ID")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save CSV files (default: data/raw/comp_<comp>_season_<season>)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing CSV files.",
    )

    args = parser.parse_args()

    out_dir = download_match_events_to_csv(
        competition_id=args.competition_id,
        season_id=args.season_id,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )

    print(f"Match events saved to: {out_dir.resolve()}")