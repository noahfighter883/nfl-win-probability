# NFL Win Probability Model

Two win probability models trained from scratch: one scores live NFL game
state play by play, the other predicts a game's winner and spread before
kickoff. [Try it Out](https://nfl-win-probability-5lzczwh2z-noahfight123.vercel.app/) (opens on a landing page explaining both and linking to
the historical replay, live tracker, and season predictions) or view the
[Case Study](https://docs.google.com/document/d/1mI910ucZKKs1_Oqxx9JjS8KDDRfjO1MkZlQfbuedTwU/edit?usp=sharing).

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
                  score live games (see "Live tracker" below), and predict upcoming games
                  before kickoff (see "Season predictions" below)
frontend/         React component + standalone HTML pages: a landing page (index.html),
                  and the historical replay, live tracker, and season predictions demos
data/             Extracted win-probability timelines for the showcase games
.github/workflows/  Scheduled jobs that poll live games and refresh season predictions
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
`frontend/replays.html` for a framework-free demo of the same thing.

`frontend/index.html` is the site's landing page (served at the domain
root) — a brief explanation of how the in-game and pregame models are each
calculated, and a clickable thumbnail linking to each of the three views
(`replays.html`, `live.html`, `season.html`). Thumbnails live in
`frontend/images/`; the live/season ones are real screenshots (captured
via headless Chrome), not mockups.

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
  showcase `games_data.json` above. It merges into whatever's already at
  `--out` rather than overwriting, and prunes anything older than 2 days —
  since ESPN's live-event list stops returning a game the instant it goes
  final, this is what keeps a just-finished game visible as a frozen result
  instead of disappearing the moment the next poll runs.
- `.github/workflows/live-poll.yml` runs that on a schedule during NFL
  windows (Thu/Sun/Mon) and publishes the result to a `live-data` branch,
  served via GitHub Pages (repo Settings → Pages → source: `live-data`) at
  `https://<user>.github.io/<repo>/live_games_data.json` — `DATA_URL` in
  `frontend/live.html` already points there. Swap it for wherever you'd
  rather host the JSON if you'd prefer running `live_score.py` on your own
  always-on server instead; the workflow is one option, not a requirement.
  Each run starts from a fresh checkout, so the workflow fetches whatever
  was last published on `live-data` first and seeds it into
  `live_score.py`'s `--out` path before scoring — without that, the
  merge/prune behavior above would have nothing to merge with.
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

## Season predictions

A separate model predicts a game's outcome *before* kickoff — win probability
and point spread — using team strength, rest, and betting-market data instead
of live game state. This is a genuinely different model from the two above
(different data source, different target, different architecture), sharing
only the project's overall conventions (temporal train/calibrate/test split,
committed small artifacts, shared feature-engineering module for train/serve
parity).

- **Data source**: nflverse's free [schedules dataset](https://github.com/nflverse/nflverse-data/releases/tag/schedules)
  (`games.csv`) — scores back to 1999, betting lines (`spread_line`,
  moneylines) for essentially every game since, rest days, and the *already-
  published* upcoming season's schedule and opening lines.
- **Team strength**: `data_pipeline/pregame_features.py` computes a from-
  scratch Elo rating per team, chronologically from 1999 (standard published
  NFL Elo methodology — margin-of-victory-adjusted K-factor, home-field bonus,
  season regression to the mean), correctly carrying ratings through team
  relocations (`OAK→LV`, `SD→LAC`, `STL→LA`) rather than resetting them.
- **Model**: `data_pipeline/train_pregame_model.py` trains a single regressor
  to predict point margin (Ridge regression won a walk-forward comparison
  against XGBoost — unsurprising at only ~5,100 training games vs. the
  in-game model's 166k plays). Win probability is *derived* from the margin
  prediction via isotonic calibration, the same technique the in-game model
  uses, so spread and win probability can never contradict each other.
- **Honest result**: an ablation comparing spread-only vs. Elo-only vs.
  combined features found the Elo/situational features did **not** clearly
  improve on the betting line alone (bootstrapped 95% CI on the RMSE
  difference included zero) — so the shipped model uses spread-line +
  situational context only (`spread_line`, `total_line`, rest, division game,
  neutral site, QB-starter-change flags), not Elo. On the 2025 holdout: model
  margin RMSE 12.29 vs. 12.29 for predicting the spread line directly; model
  win-probability AUC 0.712 vs. 0.717 for the spread-derived baseline —
  statistically indistinguishable from the market either way. This is
  reported plainly rather than oversold: **beating the closing line is hard**,
  and this project doesn't claim to.
- **Weekly scoring**: `data_pipeline/score_pregame.py` re-downloads the
  current schedule, recomputes the full Elo history fresh (cheap, no frozen
  ratings snapshot to go stale), finds the earliest week with any unplayed
  game, and writes `season_predictions.json`.

```bash
cd data_pipeline
python train_pregame_model.py                # retrains + re-evaluates (rarely needed - see below)
python score_pregame.py --out season_predictions.json  # scores the upcoming week
```
`train_pregame_model.py` downloads `games.csv` itself — no separate build
step like the in-game model's `build_dataset.py`/`train_model.py` split,
since a game-level dataset is cheap enough to build inline. Artifacts land
in `data/` (gitignored); copy the four small files it prints at the end
into `data_pipeline/pregame_model/` (committed) to ship a retrained model,
same pattern as `data_pipeline/model/` above.
- `.github/workflows/pregame-poll.yml` runs that daily (lines/rest data
  don't move nearly as fast as a live game) and publishes to the same
  `live-data` branch as the live tracker, served the same way via GitHub
  Pages. `frontend/season.html` polls it and shows the model's prediction
  directly alongside the Vegas line for comparison.

### Season predictions limitations

- Trained on **regular season only** (1999 warm-up, 2005–2022 train,
  2023–2024 calibration, 2025 holdout) — playoff predictions aren't covered.
- The model is trained on what `spread_line` represents historically
  (commonly a closing/near-closing line), but a mid-week prediction only has
  whatever's currently posted, which can still move before kickoff. The
  weekly re-run picks up newer lines as the week progresses rather than
  freezing an early snapshot.
- Feature-set selection (dropping Elo) used the same 2025 holdout as the
  final reported evaluation — a real compromise given how few NFL seasons
  exist to spare a separate selection split, not a fully clean nested holdout.

## In-game model limitations

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
