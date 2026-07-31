"""
Pregame model - weekly scorer.

Loads the persisted model + isotonic calibrator (data_pipeline/pregame_model/,
committed to the repo - not retrained here, same principle as
live_score.py's use of the already-trained in-game model), re-downloads
the current games.csv, and recomputes the full Elo/QB-continuity history
fresh each run (cheap at ~7,500 rows) so ratings reflect every completed
game so far this season - games.csv is the single source of truth, no
separate frozen ratings artifact to keep in sync.

Finds the earliest week with any unplayed game and predicts margin +
calibrated win probability for each, writing season_predictions.json:

  {
    "week": 1, "season": 2026, "generated_at": "...",
    "games": [
      {"game_id": "...", "gameday": "...", "home": "SEA", "away": "NE",
       "home_win_prob": 66.6, "predicted_margin": 4.2,
       "vegas_spread_line": 3.5, "vegas_home_moneyline": -198}
    ]
  }
"""
import argparse
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from build_pregame_dataset import load_games, completed_games
from pregame_features import compute_pregame_state, engineer_features

DEFAULT_MODEL_DIR = Path(__file__).parent / "pregame_model"


def load_artifacts(model_dir=DEFAULT_MODEL_DIR):
    model_dir = Path(model_dir)
    with open(model_dir / "pregame_meta.json") as f:
        meta = json.load(f)
    with open(model_dir / "pregame_feature_cols.json") as f:
        feature_cols = json.load(f)

    if meta["model_name"] == "ridge":
        with open(model_dir / "pregame_model.pkl", "rb") as f:
            model = pickle.load(f)
    else:
        model = xgb.XGBRegressor()
        model.load_model(model_dir / "pregame_xgb_model.json")

    # Trusted, locally-generated artifacts (produced by this repo's own
    # train_pregame_model.py, same as live_score.py already does for the
    # in-game model) - not loaded from an untrusted/external source.
    with open(model_dir / "pregame_isotonic_calibrator.pkl", "rb") as f:
        calibrator = pickle.load(f)

    return model, feature_cols, calibrator, meta


def next_upcoming_week(games_df):
    """Earliest (season, week) with any unplayed regular-season game, or
    (None, None) if the whole published schedule is already complete."""
    upcoming = games_df[games_df["home_score"].isna() & (games_df["game_type"] == "REG")]
    if upcoming.empty:
        return None, None
    row = upcoming.sort_values(["season", "week", "gameday"]).iloc[0]
    return int(row["season"]), int(row["week"])


def score_week(games_df, season, week, model, feature_cols, calibrator, meta):
    """
    Scores every *unplayed* game in the given (season, week). NFL weeks
    span several days (Thu/Sun/Mon), so by the time this runs some of the
    week's games may already be final - those are left to completed_games()
    below and excluded here, both so they don't get a stale "prediction"
    for a game whose outcome is already known, and so they aren't double-
    counted in the Elo walk (they'd otherwise appear once via the
    completed-games update path and once via this unplayed-games lookup
    path). Elo/QB state is computed across every completed game (including
    any of this week's games already played) plus this week's remaining
    unplayed games appended at the end (compute_pregame_state looks up
    their rating without performing an update, since there's no result
    yet) - so ratings reflect every result so far this season without
    needing a separate frozen ratings snapshot.
    """
    completed = completed_games(games_df)
    week_mask = ((games_df["season"] == season) & (games_df["week"] == week) &
                 (games_df["game_type"] == "REG") & games_df["home_score"].isna())
    week_games = games_df[week_mask].copy()

    combined = pd.concat([completed, week_games], ignore_index=True)
    combined = combined.sort_values(["season", "week", "gameday"]).reset_index(drop=True)
    state = compute_pregame_state(combined, hfa=meta["hfa"])

    this_week = state[(state["season"] == season) & (state["week"] == week) &
                       (state["game_type"] == "REG") & state["home_score"].isna()].copy()
    features, meta_cols = engineer_features(this_week)

    pred_margin = model.predict(features[feature_cols])
    win_prob = calibrator.predict(pred_margin)

    results = []
    for i, (_, row) in enumerate(meta_cols.iterrows()):
        entry = {
            "game_id": row["game_id"],
            "gameday": row["gameday"],
            "home": row["home_team"],
            "away": row["away_team"],
            "home_win_prob": round(float(win_prob[i]) * 100, 1),
            "predicted_margin": round(float(pred_margin[i]), 1),
        }
        spread = this_week.iloc[i]["spread_line"]
        ml = this_week.iloc[i]["home_moneyline"]
        entry["vegas_spread_line"] = float(spread) if pd.notna(spread) else None
        entry["vegas_home_moneyline"] = float(ml) if pd.notna(ml) else None
        results.append(entry)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/season_predictions.json", help="Output JSON path")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="Directory with trained model artifacts")
    args = parser.parse_args()

    model, feature_cols, calibrator, meta = load_artifacts(args.model_dir)
    games_df = load_games()

    season, week = next_upcoming_week(games_df)
    if season is None:
        print("No upcoming games found in the published schedule - nothing to score.")
        out = {"week": None, "season": None, "generated_at": datetime.now(timezone.utc).isoformat(), "games": []}
    else:
        games = score_week(games_df, season, week, model, feature_cols, calibrator, meta)
        out = {"week": week, "season": season, "generated_at": datetime.now(timezone.utc).isoformat(), "games": games}
        print(f"Scored {len(games)} games for {season} week {week}")

    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
