"""
Live win-probability scorer.

Pulls plays from live_feed.py, runs them through the same feature
engineering used for training (features.py), scores them with the
already-trained XGBoost model + isotonic calibrator (produced by
train_model.py - not retrained here), and writes a JSON file in the exact
`gamesData` shape WinProbabilityReplay.jsx already consumes:

  { [gameId]: { label, home, away, plays: [[qtr, secsLeft, homeWp, desc, down, ydstogo, fieldPosition, homeScore, awayScore], ...] } }

`secsLeft` here is `game_seconds_remaining` (counts down from 3600 across
the whole regulation game, resetting for overtime) - matching what's
already stored in the existing frontend/games_data.json showcase files, not
a per-quarter clock. The frontend's formatClock() already does `% 900` to
turn that into a quarter clock for display.

The model predicts P(posteam wins); this flips that to home-team win
probability based on which team had the ball on each play, since that's
what the frontend's `homeWp` field expects and this conversion doesn't
exist anywhere else in the project.
"""
import argparse
import json
import pickle
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import xgboost as xgb

import live_feed
from features import engineer_features, FEATURE_COLS

# poll_live_games() persists across runs (merging newly-live games into
# whatever was already there), since ESPN's "live" filter stops returning a
# game the moment it goes final - without this, a finished game would just
# vanish instead of staying visible as a frozen final result. But that same
# persistence means the file grows forever over a season unless old entries
# get pruned; this bounds it to roughly the current week's games.
MAX_GAME_AGE = timedelta(days=2)

# data_pipeline/model/ is committed to the repo (unlike data_pipeline/data/,
# which is gitignored/regenerated locally) so CI can score live plays
# without re-running build_dataset.py/train_model.py on every poll. Refresh
# it by copying the three artifact files train_model.py writes to data/
# (xgb_model.json, feature_cols.json, isotonic_calibrator.pkl) here whenever
# the model is retrained.
DEFAULT_MODEL_DIR = Path(__file__).parent / "model"


def load_artifacts(model_dir=DEFAULT_MODEL_DIR):
    model_dir = Path(model_dir)
    model = xgb.XGBClassifier()
    model.load_model(model_dir / "xgb_model.json")
    with open(model_dir / "feature_cols.json") as f:
        feature_cols = json.load(f)
    # Trusted, locally-generated artifact (produced by this repo's own
    # train_model.py, same as it already does) - not loaded from an
    # untrusted/external source.
    with open(model_dir / "isotonic_calibrator.pkl", "rb") as f:
        calibrator = pickle.load(f)
    return model, feature_cols, calibrator


def score_rows(rows, model, feature_cols, calibrator):
    """rows: list of raw play dicts from live_feed.get_game_rows.
    Returns a list of [qtr, secsLeft, homeWp, desc, down, ydstogo,
    fieldPosition, homeScore, awayScore] play tuples, in order."""
    if not rows:
        return []

    df = pd.DataFrame(rows)
    features, meta = engineer_features(df)
    features = features[feature_cols]  # enforce training column order

    raw_probs = model.predict_proba(features)[:, 1]  # P(posteam wins)
    calibrated = calibrator.predict(raw_probs)

    plays = []
    # engineer_features can drop rows (e.g. missing data); meta.index lines
    # up with features/calibrated, so use it to re-associate with the
    # original rows rather than assuming a 1:1 positional match.
    kept_rows = [rows[i] for i in meta.index]
    for row, posteam_wp in zip(kept_rows, calibrated):
        is_home_posteam = row["posteam"] == row["home_team"]
        home_wp = posteam_wp if is_home_posteam else (1 - posteam_wp)
        plays.append([
            int(row["qtr"]),
            int(row["game_seconds_remaining"]),
            round(float(home_wp) * 100, 1),
            row["desc"],
            int(row["down"]),
            int(row["ydstogo"]) if pd.notna(row["ydstogo"]) else None,
            row["field_position"],
            int(row["home_score"]),
            int(row["away_score"]),
        ])
    return plays


def build_game_label(meta, plays, home_score=None, away_score=None):
    home, away = meta["home_team"], meta["away_team"]
    status = meta.get("status", "")
    if status == "STATUS_FINAL":
        state = "Final"
    elif plays:
        qtr = plays[-1][0]
        state = f"Live · Q{qtr}" if qtr <= 4 else f"Live · OT"
    else:
        state = "Scheduled"
    return f"{away} @ {home} — {state}"


def score_event(event_id, model, feature_cols, calibrator):
    rows, meta = live_feed.get_game_rows(event_id)
    plays = score_rows(rows, model, feature_cols, calibrator)
    return {
        "label": build_game_label(meta, plays),
        "home": meta["home_team"],
        "away": meta["away_team"],
        # Raw ESPN status (e.g. "STATUS_IN_PROGRESS", "STATUS_FINAL") so the
        # frontend can show a LIVE indicator without parsing the label text.
        "status": meta.get("status"),
        "plays": plays,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }, meta


def _prune_stale(games_data, max_age=MAX_GAME_AGE):
    now = datetime.now(timezone.utc)
    kept = {}
    for game_id, entry in games_data.items():
        updated_at = entry.get("updated_at")
        if not updated_at:
            continue  # entry predates this field - safe to drop rather than keep forever
        age = now - datetime.fromisoformat(updated_at)
        if age <= max_age:
            kept[game_id] = entry
    return kept


def poll_live_games(out_path, model_dir=DEFAULT_MODEL_DIR):
    model, feature_cols, calibrator = load_artifacts(model_dir)
    live = live_feed.get_live_event_ids()

    games_data = {}
    if Path(out_path).exists():
        with open(out_path) as f:
            games_data = json.load(f)
    games_data = _prune_stale(games_data)

    for g in live:
        entry, meta = score_event(g["event_id"], model, feature_cols, calibrator)
        if entry["plays"]:
            games_data[g["event_id"]] = entry

    with open(out_path, "w") as f:
        json.dump(games_data, f)
    print(f"Wrote {len(games_data)} game(s) to {out_path}")
    return games_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", help="Score a single ESPN event ID instead of polling live games")
    parser.add_argument("--out", default="data/live_games_data.json", help="Output JSON path")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="Directory with trained model artifacts")
    args = parser.parse_args()

    model, feature_cols, calibrator = load_artifacts(args.model_dir)

    if args.event:
        entry, meta = score_event(args.event, model, feature_cols, calibrator)
        out = {args.event: entry}
        with open(args.out, "w") as f:
            json.dump(out, f)
        print(f"Scored {len(entry['plays'])} plays for {entry['away']} @ {entry['home']} -> {args.out}")
    else:
        poll_live_games(args.out, args.model_dir)
