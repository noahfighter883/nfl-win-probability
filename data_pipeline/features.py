"""
Win Probability Model - shared feature engineering.

Used by both the historical batch pipeline (build_dataset.py, operating on a
full season's worth of nflverse plays) and the live scorer (live_score.py,
operating on a single freshly-arrived play). Keeping this in one place is
what guarantees train/serve parity - the live pipeline must produce exactly
the same feature values the model was trained on.

Callers are responsible for supplying a DataFrame with the raw columns this
expects (score_differential, game_seconds_remaining, half_seconds_remaining,
down, qtr, ydstogo, yardline_100, posteam_timeouts_remaining,
defteam_timeouts_remaining, posteam_type) - for historical data these come
straight from nflverse; for live data live_feed.py builds them from ESPN's
play-by-play feed.
"""
import numpy as np
import pandas as pd

FEATURE_COLS = [
    "score_differential", "game_seconds_remaining", "yardline_100", "ydstogo",
    "posteam_timeouts_remaining", "defteam_timeouts_remaining", "score_diff_x_time",
    "is_home", "is_two_minute_drill", "is_garbage_time", "is_red_zone",
    "down_1", "down_2", "down_3", "down_4"
]

def build_label(df):
    """
    Label = 1 if the possession team (posteam) went on to win the game, else 0.
    Drops plays from tied games (result == 0), since win probability isn't
    well-defined for a game that ends in a tie.
    """
    df = df[df["result"] != 0].copy()
    # result = home_score - away_score (final). Positive -> home won.
    home_won = df["result"] > 0
    posteam_is_home = df["posteam"] == df["home_team"]
    df["label"] = np.where(posteam_is_home, home_won, ~home_won).astype(int)
    return df

def engineer_features(df):
    """
    Keep only plays with a clear offense/defense and complete situational data.
    Build the feature set for the model.
    """
    # Only plays with an actual offense on the field
    df = df[df["posteam"].notna()].copy()

    # Required fields for the model - drop rows missing any of these
    required = ["score_differential", "game_seconds_remaining", "down",
                "ydstogo", "yardline_100", "posteam_timeouts_remaining",
                "defteam_timeouts_remaining", "half_seconds_remaining", "posteam_type"]
    df = df.dropna(subset=required).copy()

    df["down"] = df["down"].astype(int)
    df["qtr"] = df["qtr"].astype(int)

    # Core features
    df["score_differential"] = df["score_differential"].astype(float)
    df["game_seconds_remaining"] = df["game_seconds_remaining"].astype(float)
    df["half_seconds_remaining"] = df["half_seconds_remaining"].astype(float)
    df["yardline_100"] = df["yardline_100"].astype(float)  # distance from opponent's endzone
    df["ydstogo"] = df["ydstogo"].astype(float)
    df["posteam_timeouts_remaining"] = df["posteam_timeouts_remaining"].astype(float)
    df["defteam_timeouts_remaining"] = df["defteam_timeouts_remaining"].astype(float)

    # Derived feature: score differential matters more as time runs out.
    # This interaction helps the model learn that a 3-pt game at 0:30 in Q4
    # is very different from a 3-pt game at 14:00 in Q1.
    df["score_diff_x_time"] = df["score_differential"] * (df["game_seconds_remaining"] / 3600.0)

    # Home-field advantage: is the possession team the home team?
    df["is_home"] = (df["posteam_type"] == "home").astype(int)

    # Two-minute drill: last 2 minutes of either half, clock management changes a lot here
    df["is_two_minute_drill"] = (df["half_seconds_remaining"] <= 120).astype(int)

    # Garbage time: late in the game with a lead too big to realistically erase
    # in the remaining possessions (more than 2 scores, i.e. >16 points, inside
    # the last 8 minutes of the 4th quarter). This is a simplified heuristic,
    # not a precise WP-based definition.
    df["is_garbage_time"] = (
        (df["qtr"] >= 4) &
        (df["game_seconds_remaining"] <= 480) &
        (df["score_differential"].abs() > 16)
    ).astype(int)

    # Red zone: offense is within 20 yards of scoring
    df["is_red_zone"] = (df["yardline_100"] <= 20).astype(int)

    # One-hot the down (down is categorical, not ordinal in a meaningful numeric sense)
    down_dummies = pd.get_dummies(df["down"], prefix="down")
    # Historical batches always contain all four downs; a single live play
    # only has one, so pad in whichever down_N columns are missing.
    for down in (1, 2, 3, 4):
        col = f"down_{down}"
        if col not in down_dummies.columns:
            down_dummies[col] = 0

    feature_cols = [
        "score_differential",
        "game_seconds_remaining",
        "yardline_100",
        "ydstogo",
        "posteam_timeouts_remaining",
        "defteam_timeouts_remaining",
        "score_diff_x_time",
        "is_home",
        "is_two_minute_drill",
        "is_garbage_time",
        "is_red_zone",
    ]

    features = pd.concat([df[feature_cols], down_dummies], axis=1)
    meta_cols = [c for c in ["game_id", "season", "posteam", "defteam", "home_team",
                              "away_team", "qtr", "desc", "label"] if c in df.columns]
    meta = df[meta_cols]

    return features, meta
