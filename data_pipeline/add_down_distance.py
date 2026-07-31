"""
One-off augmentation: adds down + distance to each play in the showcase
frontend/games_data.json, without touching the win probabilities or
descriptions already baked into that file (those were generated out-of-band
and shouldn't be recomputed here).

Matches each stored play back to its source row in the cached nflverse pbp
data by (qtr, game_seconds_remaining, desc prefix) - the stored `desc` is
truncated to 130 chars, so prefix match rather than equality. Verified this
is unambiguous for all 5 showcase games (no qtr/secsLeft group has two plays
sharing the same truncated prefix).

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
        qtr, secs, wp, desc = play
        cand = sub[(sub["qtr"] == qtr) & (sub["game_seconds_remaining"] == secs)]
        cand = cand[cand["desc"].str.startswith(desc[:DESC_PREFIX_LEN], na=False)]
        if len(cand) == 0:
            unmatched += 1
            augmented.append([qtr, secs, wp, desc, None, None])
            continue
        row = cand.iloc[0]
        down = int(row["down"]) if pd.notna(row["down"]) else None
        ydstogo = int(row["ydstogo"]) if pd.notna(row["ydstogo"]) else None
        augmented.append([qtr, secs, wp, desc, down, ydstogo])
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
