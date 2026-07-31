"""
Live NFL play-by-play feed adapter.

nflverse/nflfastR (this project's historical data source) only updates
nightly, so it can't drive a live in-game tracker. This module instead polls
ESPN's public, unauthenticated "site.api" endpoints - the same feed used by
most hobby live win-probability trackers - and maps each play into the raw
row schema `features.engineer_features` expects, so the exact same feature
math used for training also runs live.

Undocumented/unofficial API: ESPN can change these endpoints or field names
without notice. Everything ESPN-specific is isolated to this file so a
future replacement source only has to reproduce `get_live_event_ids` and
`get_game_rows`.

This module is also reused (unmodified) by backtest_live_feed.py to pull the
*same* historical games from ESPN for comparison against nflverse ground
truth. Results from that backtest (3 known OT games, run against real
nflverse pbp data): score_differential, down, ydstogo, yardline_100 and
posteam_type all match nflverse at 99-100%; timeouts at 95-100%. The
overtime time-remaining/timeout-count assumptions below held up fine - they
were not a meaningful source of mismatch. The one real, structural gap:
game_seconds_remaining/half_seconds_remaining only matched exactly 72-89%
of the time, off by anywhere from a few seconds to ~30-90s on the rest,
spread across regulation as well as OT - this looks like ESPN's logged play
clock drifting from the NFL's official gamebook clock nflverse uses, not a
bug in the formula here. A few seconds of noise on a continuous feature is
unlikely to move predictions meaningfully, but it can occasionally flip a
play across the two-minute-drill/garbage-time thresholds, which are hard
0/1 cutoffs - re-run backtest_live_feed.py after any change here to confirm
this hasn't regressed.
"""
import re
import time

import requests

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary"

REQUEST_TIMEOUT = 10

# Plays with one of these types never have a meaningful down/distance and
# are excluded from the feature rows, same as build_dataset.py's dropna on
# `down`. They're still walked for score/timeout bookkeeping.
NO_DOWN_PLAY_TYPES = {
    "Kickoff", "Kickoff Return (Offense)", "Timeout", "Two-minute warning",
    "End Period", "End of Half", "End of Regulation", "End of Game",
    "Extra Point Good", "Extra Point Missed", "Two-Point Conversion Good",
    "Two-Point Conversion Missed", "Penalty", "Coin Toss",
}

TIMEOUT_RE = re.compile(r"Timeout #\d+ by (\w+)")

# Timeouts available at the start of each half of regulation, and at the
# start of a regular-season overtime period (playoff OT uses full quarters
# and normal timeout rules - not handled here since this project only
# covers the regular season). Backtested against nflverse ground truth for
# 3 known OT games (see backtest_live_feed.py) - matched 95-100%.
TIMEOUTS_PER_REGULATION_HALF = 3
TIMEOUTS_PER_REGULAR_SEASON_OT = 2


def _get(url, **params):
    resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def get_live_event_ids():
    """Returns [{"event_id", "home": abbr, "away": abbr, "label"}] for games currently in progress."""
    data = _get(SCOREBOARD_URL)
    live = []
    for event in data.get("events", []):
        state = event.get("status", {}).get("type", {}).get("state")
        if state != "in":
            continue
        comp = event["competitions"][0]
        home = next(c for c in comp["competitors"] if c["homeAway"] == "home")
        away = next(c for c in comp["competitors"] if c["homeAway"] == "away")
        live.append({
            "event_id": event["id"],
            "home": home["team"]["abbreviation"],
            "away": away["team"]["abbreviation"],
            "label": event.get("shortName", ""),
        })
    return live


def _team_map(summary):
    """team_id -> {"abbr": ..., "home_away": "home"/"away"}"""
    comp = summary["header"]["competitions"][0]
    return {
        c["team"]["id"]: {"abbr": c["team"]["abbreviation"], "home_away": c["homeAway"]}
        for c in comp["competitors"]
    }


def _ordered_plays(summary):
    plays = []
    for drive in summary.get("drives", {}).get("previous", []):
        plays.extend(drive.get("plays", []))
    current = summary.get("drives", {}).get("current")
    if current:
        plays.extend(current.get("plays", []))
    plays.sort(key=lambda p: int(p["sequenceNumber"]))
    return plays


def _clock_seconds(play):
    mm, ss = play["clock"]["displayValue"].split(":")
    return int(mm) * 60 + int(ss)


def _time_remaining(period, clock_secs):
    """Returns (game_seconds_remaining, half_seconds_remaining)."""
    if period <= 4:
        game_secs = clock_secs + (4 - period) * 900
        if period in (1, 2):
            half_secs = clock_secs + (2 - period) * 900
        else:
            half_secs = clock_secs + (4 - period) * 900
        return game_secs, half_secs
    # Overtime: treated as its own self-contained period. Backtested against
    # nflverse ground truth (see module docstring) - this convention wasn't
    # a notable source of mismatch.
    return clock_secs, clock_secs


def _timeout_limit(period):
    return TIMEOUTS_PER_REGULATION_HALF if period <= 4 else TIMEOUTS_PER_REGULAR_SEASON_OT


def get_game_rows(event_id):
    """
    Fetches one game's live/final play-by-play from ESPN and returns
    (rows, meta) where `rows` is a list of dicts matching the raw schema
    `features.engineer_features` consumes, and `meta` describes the game.
    """
    summary = _get(SUMMARY_URL, event=event_id)
    team_map = _team_map(summary)
    comp = summary["header"]["competitions"][0]
    home = next(c for c in comp["competitors"] if c["homeAway"] == "home")
    away = next(c for c in comp["competitors"] if c["homeAway"] == "away")
    home_abbr, away_abbr = home["team"]["abbreviation"], away["team"]["abbreviation"]

    meta = {
        "game_id": event_id,
        "season": summary["header"].get("season", {}).get("year"),
        "home_team": home_abbr,
        "away_team": away_abbr,
        "status": comp["status"]["type"]["name"],
    }

    running_score = {home["team"]["id"]: 0, away["team"]["id"]: 0}
    timeouts = {home["team"]["id"]: 3, away["team"]["id"]: 3}
    last_half_bucket = None  # (1,2) -> "first_half", (3,4) -> "second_half", 5+ -> "ot"

    rows = []
    for play in _ordered_plays(summary):
        period = play["period"]["number"]
        half_bucket = "first_half" if period <= 2 else ("second_half" if period <= 4 else "ot")
        if half_bucket != last_half_bucket:
            limit = _timeout_limit(period)
            timeouts = {tid: limit for tid in timeouts}
            last_half_bucket = half_bucket

        play_type = play["type"]["text"]

        if play_type == "Timeout":
            m = TIMEOUT_RE.search(play.get("text", ""))
            if m:
                abbr = m.group(1)
                for tid, info in team_map.items():
                    if info["abbr"] == abbr and tid in timeouts:
                        timeouts[tid] = max(0, timeouts[tid] - 1)

        if play_type not in NO_DOWN_PLAY_TYPES:
            offense = next((p for p in play.get("teamParticipants", []) if p["type"] == "offense"), None)
            defense = next((p for p in play.get("teamParticipants", []) if p["type"] == "defense"), None)
            start = play.get("start", {})
            down = start.get("down", 0)

            if offense and defense and down in (1, 2, 3, 4):
                pos_id, def_id = offense["id"], defense["id"]
                pos_score = running_score.get(pos_id, 0)
                def_score = running_score.get(def_id, 0)
                game_secs, half_secs = _time_remaining(period, _clock_seconds(play))
                pos_info = team_map.get(pos_id, {})
                def_info = team_map.get(def_id, {})

                rows.append({
                    "game_id": event_id,
                    "season": meta["season"],
                    "posteam": pos_info.get("abbr"),
                    "defteam": def_info.get("abbr"),
                    "home_team": home_abbr,
                    "away_team": away_abbr,
                    "qtr": period,
                    "desc": play.get("text", ""),
                    "score_differential": pos_score - def_score,
                    "game_seconds_remaining": game_secs,
                    "half_seconds_remaining": half_secs,
                    "down": down,
                    "ydstogo": start.get("distance"),
                    "yardline_100": start.get("yardsToEndzone"),
                    "posteam_timeouts_remaining": timeouts.get(pos_id, 0),
                    "defteam_timeouts_remaining": timeouts.get(def_id, 0),
                    "posteam_type": pos_info.get("home_away"),
                    # Display-only, not a model feature: score entering this
                    # play, same "before" convention as score_differential above.
                    "home_score": running_score.get(home["team"]["id"], 0),
                    "away_score": running_score.get(away["team"]["id"], 0),
                })

        # Post-play cumulative score becomes the "before" score for the next play.
        running_score[home["team"]["id"]] = play.get("homeScore", running_score[home["team"]["id"]])
        running_score[away["team"]["id"]] = play.get("awayScore", running_score[away["team"]["id"]])

    return rows, meta


if __name__ == "__main__":
    live = get_live_event_ids()
    if not live:
        print("No live games right now.")
    for g in live:
        print(g["event_id"], g["away"], "@", g["home"], "-", g["label"])
        rows, meta = get_game_rows(g["event_id"])
        print(f"  {len(rows)} scoreable plays so far, status={meta['status']}")
        time.sleep(0.5)
