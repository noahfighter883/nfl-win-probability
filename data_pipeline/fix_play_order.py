"""
One-off fix: re-sorts each showcase game's play list into true chronological
order.

frontend/games_data.json's plays were apparently originally sorted by
game_seconds_remaining alone, ignoring qtr. Q4's clock (900 -> 0) and OT's
clock (600 -> 0) cover overlapping numeric ranges despite representing
completely different points in the game, so a secsLeft-only sort interleaves
OT plays into the middle of the 4th quarter (and similarly for any other
quarter boundary) - verified present in all 5 showcase games (7-21 misordered
plays each out of ~150-195).

Fix: stable sort by (qtr ascending, game_seconds_remaining descending) -
qtr always groups plays into the right period, and within a period the
clock only counts down. Stable sort preserves original relative order for
any exact (qtr, secsLeft) ties.

Run once from data_pipeline/ if games_data.json is regenerated from scratch
- not part of the regular data pipeline.
"""
import json
from pathlib import Path

GAMES_DATA_PATH = Path(__file__).parent.parent / "frontend" / "games_data.json"


def count_violations(plays):
    violations = 0
    for i in range(1, len(plays)):
        pq, ps = plays[i - 1][0], plays[i - 1][1]
        q, s = plays[i][0], plays[i][1]
        if q < pq or (q == pq and s > ps):
            violations += 1
    return violations


def main():
    with open(GAMES_DATA_PATH) as f:
        games = json.load(f)

    for game_id, g in games.items():
        before = count_violations(g["plays"])
        g["plays"].sort(key=lambda p: (p[0], -p[1]))
        after = count_violations(g["plays"])
        print(f"{game_id}: {before} -> {after} order violations")

    with open(GAMES_DATA_PATH, "w") as f:
        json.dump(games, f)
    print(f"Wrote {GAMES_DATA_PATH}")


if __name__ == "__main__":
    main()
