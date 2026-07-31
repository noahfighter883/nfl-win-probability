"""
Pregame model - dataset builder.

Downloads nflverse's schedules/games dataset, computes Elo ratings +
QB-continuity flags across every known-outcome game (regular season and
playoffs - playoff results are real signal for team strength, even though
only regular-season rows become labeled training examples below), and
engineers the pregame feature set.

Unlike build_dataset.py's historical pbp_<year>.csv.gz files (static once a
season is final, safe to cache indefinitely), games.csv changes weekly
in-season - lines fill in, rest days become known, scores get added - so
this always re-downloads rather than reusing a cached copy.
"""
import os
import urllib.request

import numpy as np
import pandas as pd

from pregame_features import compute_pregame_state, engineer_features, HOME_FIELD_ADVANTAGE

GAMES_URL = "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv"


def download_games():
    os.makedirs("data", exist_ok=True)
    path = "data/games.csv"
    print("Downloading games.csv (always fresh - this file changes weekly in-season)...")
    urllib.request.urlretrieve(GAMES_URL, path)
    return path


def load_games():
    path = download_games()
    df = pd.read_csv(path, low_memory=False)
    return df


def build_label(df):
    """
    Label = margin = home_score - away_score. Drops tied games, same
    reasoning as features.py's build_label for the in-game model (win
    probability isn't well-defined for a tie).
    """
    df = df[df["home_score"] != df["away_score"]].copy()
    df["margin"] = df["home_score"] - df["away_score"]
    return df


def completed_games(games_df):
    """All games with a known result, sorted chronologically. `games_df` is
    the raw games.csv frame (e.g. from load_games())."""
    completed = games_df[games_df["home_score"].notna() & games_df["away_score"].notna()].copy()
    return completed.sort_values(["season", "week", "gameday"]).reset_index(drop=True)


def build_labeled_dataset(games_df, hfa=HOME_FIELD_ADVANTAGE, min_season=2005):
    """
    Runs the full pregame pipeline (Elo/QB state -> feature engineering ->
    labeling) for a given home-field-advantage value, without touching
    disk. Exposed separately from main() so train_pregame_model.py can call
    this repeatedly while sweeping candidate `hfa` values during tuning,
    reusing the exact same logic the final persisted dataset is built with.

    Elo/QB-continuity state is computed across every completed game (REG +
    playoffs, from 1999 on) so ratings reflect the fullest signal
    available; the returned frame is then restricted to labeled rows:
    regular season only (playoffs are a systematically different
    population - only good teams reach them, atypical rest patterns),
    `min_season` onward (default 2005, giving Elo 1999-2004 to warm up),
    ties dropped (same reasoning as features.py's build_label).
    """
    completed = completed_games(games_df)
    state = compute_pregame_state(completed, hfa=hfa)

    labeled = state[(state["game_type"] == "REG") & (state["season"] >= min_season)].copy()
    labeled = build_label(labeled)

    features, meta = engineer_features(labeled)
    return pd.concat([meta.reset_index(drop=True), features.reset_index(drop=True)], axis=1)


def main():
    print("Loading schedule/results data...")
    df = load_games()
    dataset = build_labeled_dataset(df)
    print(f"Labeled dataset (REG season >= 2005, ties dropped): {len(dataset)} games")
    print(f"Feature columns: {[c for c in dataset.columns if c not in ('game_id','season','week','game_type','gameday','home_team','away_team','home_score','away_score','margin','home_moneyline','away_moneyline')]}")

    dataset.to_parquet("data/pregame_dataset.parquet", index=False)
    print("Saved to data/pregame_dataset.parquet")
    print()
    print("Home win rate:", (dataset["margin"] > 0).mean())
    print("Games per season:")
    print(dataset.groupby("season").size())


if __name__ == "__main__":
    main()
