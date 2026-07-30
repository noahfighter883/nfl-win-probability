# NFL Win Probability Model

A win probability model trained from scratch on play-by-play NFL data, plus
an interactive replay of five of the most dramatic games of the last four
seasons. [Try it Out](https://nfl-win-probability-5lzczwh2z-noahfight123.vercel.app/) or view the [Case Study](https://docs.google.com/document/d/1mI910ucZKKs1_Oqxx9JjS8KDDRfjO1MkZlQfbuedTwU/edit?usp=sharing).

## What this is

Given the game situation on any play (score, time remaining, field position,
down and distance, timeouts), the model estimates the probability that the
team on offense goes on to win. It's trained on ~166k plays from the
2021–2024 seasons.

- **Model**: XGBoost classifier, isotonic-calibrated
- **Performance**: 0.845 AUC on a held-out 2024 season
- **Data source**: [nflverse](https://github.com/nflverse/nflverse-data) play-by-play data

## Repo structure

```
data_pipeline/    Python scripts: build the training dataset, train + calibrate the model,
                  and score live games (see "Live tracker" below)
frontend/         React component + standalone HTML demos for the historical replay and
                  the live tracker
data/             Extracted win-probability timelines for the showcase games
.github/workflows/  Scheduled job that polls live games during NFL windows
```

## Data pipeline

```bash
cd data_pipeline
pip install -r requirements.txt
python build_dataset.py   # downloads pbp data, builds labels + features
python train_model.py     # trains, calibrates, evaluates
```

`build_dataset.py` downloads four seasons of play-by-play data from
nflverse, constructs the win/loss label per play, and engineers features
including score differential, a score-differential × time-remaining
interaction, red zone, two-minute drill, garbage time, and home-field
indicators.

`train_model.py` trains on 2021–2022, fits isotonic calibration on 2023
(kept separate from training to avoid leakage), and reports final metrics
on the untouched 2024 holdout. It writes model artifacts to `data/` (git-
ignored, regenerated locally). `data_pipeline/model/` holds a copy of the
three small artifact files (`xgb_model.json`, `feature_cols.json`,
`isotonic_calibrator.pkl`) that *is* committed, so the live tracker below
can score plays without re-running the pipeline — refresh it manually by
copying from `data/` after retraining.

## Frontend

`frontend/WinProbabilityReplay.jsx` is a self-contained React component
(no charting library — hand-built SVG) that replays five historic games
play by play against their model-computed win probability curve. See
`frontend/index.html` for a framework-free demo of the same thing.

## Live tracker

The same trained model also scores NFL games as they're actually being
played, instead of only replaying five fixed historical games.

nflverse (the historical data source above) only updates nightly, so it
can't drive anything live. Instead:

- `data_pipeline/live_feed.py` polls ESPN's public, unauthenticated
  `site.api.espn.com` endpoints for plays as they happen, and maps them
  into the same raw schema `features.py` expects (score, time, down,
  distance, field position, timeouts).
- `data_pipeline/live_score.py` runs those plays through the same feature
  engineering used for training, scores them with the already-trained
  model (`data_pipeline/model/` — committed to the repo, not retrained per
  poll), and writes a `live_games_data.json` in the same shape as the
  showcase `games_data.json` above.
- `.github/workflows/live-poll.yml` runs that on a schedule during NFL
  windows (Thu/Sun/Mon) and publishes the result to a `live-data` branch.
  Point a static host (e.g. GitHub Pages) at that branch and set `DATA_URL`
  in `frontend/live.html` to it — or run `live_score.py` yourself on any
  always-on server and serve the JSON however you like; the workflow is
  one option, not a requirement.
- `frontend/live.html` and `WinProbabilityReplay.jsx`'s `liveUrl` prop poll
  that JSON and render the same replay UI, with a pulsing LIVE badge on
  in-progress games.
- `data_pipeline/backtest_live_feed.py` compares ESPN-derived plays against
  real nflverse pbp data for known historical games, to catch mapping bugs
  before trusting the live feed. See `live_feed.py`'s module docstring for
  the last backtest's results.

### Live tracker limitations

- ESPN's API is public but unofficial/undocumented — it can change without
  notice. Everything ESPN-specific is isolated in `live_feed.py`.
- `game_seconds_remaining` derived from ESPN's play clock doesn't always
  match nflverse's official gamebook clock exactly — typically off by a
  few seconds, occasionally by up to ~30–90s (backtested against 3 known
  games; see `live_feed.py`). This is unlikely to move predictions
  meaningfully on its own, but could occasionally flip a play across the
  two-minute-drill/garbage-time thresholds.
- Timeouts remaining are reconstructed from the play-by-play feed itself
  (watching for "Timeout" plays), not read from an authoritative live
  field — backtested at 95–100% agreement with nflverse.

## Known limitations

- The model is noticeably conservative at the extreme low end of win
  probability (large-deficit, late-game situations) — actual comeback rates
  from a 15+ point deficit are lower than the model predicts, even after
  isotonic calibration. This seems to be a data-sparsity issue: extreme
  blowout comebacks are rare enough that there isn't much signal for the
  calibrator to learn from at that end of the distribution.
- Of the five showcase games, only the 2024 game was a genuine holdout
  the model never saw during training or calibration. The others were
  chosen for their historical significance, not to demonstrate held-out
  performance.
