"""
Backtest: compare ESPN-derived play state against nflverse ground truth for
known completed games.

live_feed.py's ESPN parsing makes a couple of assumptions that aren't
confirmed against real data (see its module docstring): the overtime
time-remaining convention, and OT timeout counts. This script catches those
- plus any other mapping bugs (down/distance/field position errors,
home/away flips) - by pulling the same games from both live_feed.py (ESPN)
and the local nflverse pbp CSVs (ground truth, same source build_dataset.py
uses) and diffing the raw fields that feed features.py.

Requires data_pipeline/data/pbp_<year>.csv.gz to exist for each season
covered below (run build_dataset.py first, which downloads them).

Usage:
    python backtest_live_feed.py
"""
import sys
from pathlib import Path

import pandas as pd

import live_feed
from features import FEATURE_COLS

DATA_DIR = Path(__file__).parent / "data"

# (nflverse game_id, ESPN scoreboard date YYYYMMDD, home abbr, away abbr).
# Three known overtime games - including two of the five games in this
# project's own showcase replay - chosen specifically to exercise the OT
# time/timeout assumptions live_feed.py has to guess at.
KNOWN_GAMES = [
    ("2022_15_IND_MIN", "20221217", "MIN", "IND"),
    ("2023_22_SF_KC", "20240211", "KC", "SF"),
    ("2021_20_BUF_KC", "20220123", "KC", "BUF"),
]

RAW_COLS = [
    "score_differential", "game_seconds_remaining", "half_seconds_remaining",
    "down", "ydstogo", "yardline_100",
    "posteam_timeouts_remaining", "defteam_timeouts_remaining", "posteam_type",
]

MATCH_KEY_COLS = ["qtr", "down", "ydstogo", "yardline_100"]


def _wsh_alias(abbr):
    return {"WAS": "WSH", "WSH": "WAS"}.get(abbr, abbr)


def find_espn_event(date, home, away):
    data = live_feed._get(live_feed.SCOREBOARD_URL, dates=date)
    for event in data.get("events", []):
        comp = event["competitions"][0]
        h = next(c for c in comp["competitors"] if c["homeAway"] == "home")
        a = next(c for c in comp["competitors"] if c["homeAway"] == "away")
        h_abbr, a_abbr = h["team"]["abbreviation"], a["team"]["abbreviation"]
        if h_abbr in (home, _wsh_alias(home)) and a_abbr in (away, _wsh_alias(away)):
            return event["id"]
    return None


def load_nflverse_rows(game_id, season):
    path = DATA_DIR / f"pbp_{season}.csv.gz"
    if not path.exists():
        print(f"  Missing {path} - run build_dataset.py first. Skipping.")
        return None
    df = pd.read_csv(path, compression="gzip", low_memory=False)
    df = df[df["game_id"] == game_id]
    df = df[df["posteam"].notna()]
    df = df.dropna(subset=[c for c in RAW_COLS + ["qtr"] if c != "posteam_type"] + ["posteam_type"])
    df["down"] = df["down"].astype(int)
    df["qtr"] = df["qtr"].astype(int)
    cols = list(dict.fromkeys(MATCH_KEY_COLS + RAW_COLS))  # qtr/down/ydstogo/yardline_100 appear in both
    return df[cols].to_dict("records")


def align(nflverse_rows, espn_rows):
    """Greedy sequential alignment on (qtr, down, ydstogo, yardline_100),
    with a small lookahead to resync past occasional insertions/deletions
    (e.g. a penalty-only play one source keeps and the other drops)."""
    def key(r):
        return tuple(r[c] for c in MATCH_KEY_COLS)

    i, j = 0, 0
    matched = []
    unmatched = 0
    while i < len(nflverse_rows) and j < len(espn_rows):
        a, b = nflverse_rows[i], espn_rows[j]
        if key(a) == key(b):
            matched.append((a, b))
            i += 1
            j += 1
            continue
        resynced = False
        for lookahead in range(1, 4):
            if i + lookahead < len(nflverse_rows) and key(nflverse_rows[i + lookahead]) == key(b):
                i += lookahead
                resynced = True
                break
            if j + lookahead < len(espn_rows) and key(a) == key(espn_rows[j + lookahead]):
                j += lookahead
                resynced = True
                break
        if not resynced:
            unmatched += 1
            i += 1
            j += 1
    return matched, unmatched


def report(matched, unmatched, n_a, n_b):
    print(f"  nflverse plays: {n_a} | ESPN plays: {n_b} | matched: {len(matched)} | unresynced: {unmatched}")
    if not matched:
        print("  No plays matched at all - something is fundamentally broken in the mapping.")
        return

    for col in RAW_COLS:
        agree = sum(1 for a, b in matched if a[col] == b[col])
        pct = 100 * agree / len(matched)
        flag = "" if pct > 98 else "  <-- CHECK THIS"
        print(f"    {col:32s} exact match: {pct:5.1f}%{flag}")

    mismatches = [(a, b) for a, b in matched if a["game_seconds_remaining"] != b["game_seconds_remaining"]]
    if mismatches:
        print(f"  Sample game_seconds_remaining mismatches (nflverse vs ESPN), qtr>=5 highlighted for OT:")
        for a, b in mismatches[:5]:
            ot = " [OT]" if a["qtr"] >= 5 else ""
            print(f"    qtr={a['qtr']}{ot} down={a['down']} ydstogo={a['ydstogo']}: "
                  f"nflverse={a['game_seconds_remaining']} espn={b['game_seconds_remaining']}")


def main():
    for game_id, date, home, away in KNOWN_GAMES:
        season = int(game_id.split("_")[0])
        print(f"\n{game_id} ({away} @ {home}, {date})")

        nflverse_rows = load_nflverse_rows(game_id, season)
        if nflverse_rows is None:
            continue

        event_id = find_espn_event(date, home, away)
        if not event_id:
            print(f"  Could not find an ESPN event for {away} @ {home} on {date}. Skipping.")
            continue

        espn_rows, _ = live_feed.get_game_rows(event_id)

        matched, unmatched = align(nflverse_rows, espn_rows)
        report(matched, unmatched, len(nflverse_rows), len(espn_rows))


if __name__ == "__main__":
    sys.exit(main())
