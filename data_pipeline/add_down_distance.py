"""
One-off augmentation: adds down + distance and the running score to each
play in the showcase frontend/games_data.json, and restores the full
(untruncated) play description - the stored `desc` was truncated to 130
chars by whatever out-of-band process originally generated this file. Win
probabilities are left untouched (those aren't recomputed here).

Score is the score *after* this play resolves (posteam/defteam_score_post,
remapped to home/away) - deliberately the "after" convention, unlike
live_score.py's "before" convention for the same play, since these
descriptions already narrate the play's outcome (e.g. "...TOUCHDOWN") and
showing the pre-play score next to that would read as wrong.

Matches each stored play back to its source row in the cached nflverse pbp
data by (qtr, game_seconds_remaining, desc prefix). Idempotent - re-running
against an already-augmented file works the same as against the original
4-element tuples, since matching only ever looks at the first 4 fields.

Run once from data_pipeline/ after games_data.json changes (new showcase
games added, etc.) - not part of the regular data pipeline.
"""
import json
from pathlib import Path

import pandas as pd

GAMES_DATA_PATH = Path(__file__).parent.parent / "frontend" / "games_data.json"
DESC_PREFIX_LEN = 100  # shorter than the 130-char truncation, safely unique


def load_pbp(year):
    path = Path(__file__).parent / "data" / f"pbp_{year}.csv.gz"
    return pd.read_csv(path, compression="gzip", low_memory=False)


def augment_game(game_id, plays):
    year = game_id.split("_")[0]
    df = load_pbp(year)
    sub = df[df["game_id"] == game_id]

    augmented = []
    unmatched = 0
    for play in plays:
        qtr, secs, wp, desc = play[:4]
        cand = sub[(sub["qtr"] == qtr) & (sub["game_seconds_remaining"] == secs)]
        cand = cand[cand["desc"].str.startswith(desc[:DESC_PREFIX_LEN], na=False)]
        if len(cand) == 0:
            unmatched += 1
            augmented.append([qtr, secs, wp, desc, None, None])
            continue
        row = cand.iloc[0]
        down = int(row["down"]) if pd.notna(row["down"]) else None
        ydstogo = int(row["ydstogo"]) if pd.notna(row["ydstogo"]) else None
        is_home_posteam = row["posteam"] == row["home_team"]
        home_score = row["posteam_score_post"] if is_home_posteam else row["defteam_score_post"]
        away_score = row["defteam_score_post"] if is_home_posteam else row["posteam_score_post"]
        augmented.append([
            qtr, secs, wp, row["desc"], down, ydstogo,
            int(home_score) if pd.notna(home_score) else None,
            int(away_score) if pd.notna(away_score) else None,
        ])
    if unmatched:
        print(f"  {game_id}: {unmatched}/{len(plays)} plays unmatched (left without down/distance)")
    return augmented


def main():
    with open(GAMES_DATA_PATH) as f:
        games = json.load(f)

    for game_id, g in games.items():
        print(f"Augmenting {game_id} ({len(g['plays'])} plays)...")
        g["plays"] = augment_game(game_id, g["plays"])

    with open(GAMES_DATA_PATH, "w") as f:
        json.dump(games, f)
    print(f"Wrote {GAMES_DATA_PATH}")


if __name__ == "__main__":
    main()
