# NFL Win Probability Model

A win probability model trained from scratch on play-by-play NFL data, plus
an interactive replay of five of the most dramatic games of the last four
seasons. [Try it Out](https://nfl-win-probability-5lzczwh2z-noahfight123.vercel.app/) Or view the [Case Study](https://docs.google.com/document/d/1mI910ucZKKs1_Oqxx9JjS8KDDRfjO1MkZlQfbuedTwU/edit?usp=sharing).

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
data_pipeline/    Python scripts: build the training dataset, train + calibrate the model
frontend/         React component + standalone HTML demo for the game replay
data/             Extracted win-probability timelines for the showcase games
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
on the untouched 2024 holdout.

## Frontend

`frontend/WinProbabilityReplay.jsx` is a self-contained React component
(no charting library — hand-built SVG) that replays five historic games
play by play against their model-computed win probability curve. See
`frontend/preview.html` for a framework-free demo of the same thing.

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
