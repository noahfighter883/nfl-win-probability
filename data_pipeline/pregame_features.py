"""
Pregame win probability / spread model - shared Elo ratings + feature
engineering.

Elo methodology follows the standard published NFL Elo approach
(FiveThirtyEight/Neil Paine): margin-of-victory-adjusted K-factor updates,
a home-field bonus, and season-boundary regression to the mean. Computed
chronologically across every game (regular season + playoffs - playoff
results are real signal for *ratings*, even though the model is only
trained/evaluated on regular-season rows) from 1999 onward, producing each
team's *pregame* rating entering every game they play.

This module is the single source of truth for Elo and QB-continuity
tracking, used by both the batch trainer (build_pregame_dataset.py) and the
weekly scorer (score_pregame.py), so ratings are always computed the same
way and never drift out of sync between training and live prediction -
same train/serve parity principle as features.py for the in-game model.
"""
import numpy as np
import pandas as pd

START_RATING = 1500.0
EXPANSION_RATING = 1300.0  # a team's true first-ever season starts here (e.g. Houston in 2002)
K = 20.0
HOME_FIELD_ADVANTAGE = 55.0  # default; train_pregame_model.py may tune this against the calibration set
REGRESSION_FACTOR = 2.0 / 3.0  # fraction of a team's rating *kept* across a season boundary

# Franchises that relocated/rebranded - ratings carry over as one continuous
# history under the new code, never reset. Verified against nflverse's
# games.csv: OAK->LV (last OAK season 2019, first LV season 2020),
# SD->LAC (2016->2017), STL->LA (2015->2016). WAS needs no entry despite the
# Redskins/Football Team/Commanders renames - the CSV code never changed.
TEAM_ALIASES = {"OAK": "LV", "SD": "LAC", "STL": "LA"}

FEATURE_COLS = [
    "spread_line", "total_line", "home_rest", "away_rest", "rest_diff",
    "div_game", "is_neutral_site", "home_elo", "away_elo", "elo_diff",
    "qb_change_home", "qb_change_away",
]


def canonical_team(team):
    return TEAM_ALIASES.get(team, team)


def compute_pregame_state(games, hfa=HOME_FIELD_ADVANTAGE, k=K):
    """
    games: DataFrame with one row per game, sorted chronologically
    (season, week, gameday - week increases monotonically through playoffs
    in this dataset, so this sort is chronological within a season too).
    Rows may be unplayed (home_score/away_score null, e.g. an upcoming
    week) as long as every *played* game sorts before any game that
    depends on its result - unplayed rows get a rating lookup only (no Elo
    update, since there's no result yet), which is exactly what
    score_pregame.py needs to rate next week's games off of every
    completed game so far without a separate frozen ratings artifact. Ties
    are fine here (only build_pregame_dataset.py's labeled training rows
    need to drop them, same reasoning as features.py's build_label).

    Returns a copy of `games` with four added columns: home_elo, away_elo
    (each team's rating *entering* that game, before it updates anything),
    and qb_change_home, qb_change_away (1 if that team's starter differs
    from their immediately preceding game, else 0; 0 for a team's first
    game in the dataset, or when a starter isn't known yet, since there's
    nothing to compare against).
    """
    ratings = {}
    last_qb = {}
    current_season = None

    home_elo = np.empty(len(games))
    away_elo = np.empty(len(games))
    qb_change_home = np.empty(len(games), dtype=int)
    qb_change_away = np.empty(len(games), dtype=int)

    def get_rating(team, season):
        canon = canonical_team(team)
        if canon not in ratings:
            ratings[canon] = EXPANSION_RATING if season > 1999 else START_RATING
        return ratings[canon]

    def get_qb_change(team, qb_name):
        canon = canonical_team(team)
        prev = last_qb.get(canon)
        changed = int(prev is not None and bool(qb_name) and prev != qb_name)
        if qb_name:
            last_qb[canon] = qb_name
        return changed

    for i, row in enumerate(games.itertuples()):
        if current_season is not None and row.season != current_season:
            # Season boundary: regress every known team's rating toward 1500
            # (roster turnover means last season's rating shouldn't carry
            # forward at full strength).
            for team in ratings:
                ratings[team] = START_RATING + (ratings[team] - START_RATING) * REGRESSION_FACTOR
        current_season = row.season

        home, away = canonical_team(row.home_team), canonical_team(row.away_team)
        home_r = get_rating(row.home_team, row.season)
        away_r = get_rating(row.away_team, row.season)
        home_elo[i] = home_r
        away_elo[i] = away_r

        qb_change_home[i] = get_qb_change(row.home_team, getattr(row, "home_qb_name", None))
        qb_change_away[i] = get_qb_change(row.away_team, getattr(row, "away_qb_name", None))

        if pd.isna(row.home_score) or pd.isna(row.away_score):
            continue  # unplayed game: rating looked up above, nothing to update from

        game_hfa = 0.0 if row.location == "Neutral" else hfa
        expected_home = 1.0 / (1.0 + 10 ** (-((home_r + game_hfa) - away_r) / 400.0))

        margin = row.home_score - row.away_score
        actual_home = 1.0 if margin > 0 else (0.0 if margin < 0 else 0.5)

        # Margin-of-victory multiplier: dampens rating credit for a blowout
        # the winner was already expected to produce, amplifies it for an
        # upset blowout. Ties give ln(0+1)=0 -> zero rating change under
        # this standard formula - an accepted quirk (ties are rare: 15 in
        # 27 seasons in this dataset), not a bug.
        winner_elo_diff = (home_r + game_hfa - away_r) if margin >= 0 else (away_r - (home_r + game_hfa))
        mov_mult = np.log(abs(margin) + 1) * (2.2 / (0.001 * winner_elo_diff + 2.2))

        delta = k * mov_mult * (actual_home - expected_home)
        ratings[home] = home_r + delta
        ratings[away] = away_r - delta

    out = games.copy()
    out["home_elo"] = home_elo
    out["away_elo"] = away_elo
    out["qb_change_home"] = qb_change_home
    out["qb_change_away"] = qb_change_away
    return out


def engineer_features(games_with_state):
    """
    games_with_state: output of compute_pregame_state (or a subset of its
    rows/columns with the same schema). Derives the final feature columns
    and returns (features, meta) - same two-frame split as features.py.
    """
    df = games_with_state.copy()

    df["rest_diff"] = df["home_rest"].astype(float) - df["away_rest"].astype(float)
    df["is_neutral_site"] = (df["location"] == "Neutral").astype(int)
    df["div_game"] = df["div_game"].astype(int)
    df["elo_diff"] = df["home_elo"] - df["away_elo"]
    df["spread_line"] = df["spread_line"].astype(float)
    df["total_line"] = df["total_line"].astype(float)
    df["home_rest"] = df["home_rest"].astype(float)
    df["away_rest"] = df["away_rest"].astype(float)

    features = df[FEATURE_COLS]
    meta_cols = [c for c in ["game_id", "season", "week", "game_type", "gameday",
                              "home_team", "away_team", "home_score", "away_score",
                              "margin", "home_moneyline", "away_moneyline"] if c in df.columns]
    meta = df[meta_cols]
    return features, meta
