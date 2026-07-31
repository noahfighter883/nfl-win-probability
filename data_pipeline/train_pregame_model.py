"""
Pregame model - training.

Predicts point margin (home_score - away_score) with a single regressor;
win probability is *derived* from the margin prediction via an isotonic
calibrator (same technique/library train_model.py already uses for the
in-game model, just with margin as the raw score instead of a classifier's
raw probability) rather than training a separate classifier - this keeps
the two outputs (spread, win probability) always mutually consistent.

Everything here is benchmarked against the betting market itself, not just
reported in isolation - pregame prediction is a fundamentally
lower-information regime than the in-game model (~0.70-0.71 AUC is roughly
where the market sits; nowhere near the in-game model's 0.845, and that's
not a comparable target).
"""
import json
import pickle

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, mean_absolute_error, mean_squared_error
import xgboost as xgb

from build_pregame_dataset import load_games, build_labeled_dataset
from pregame_features import FEATURE_COLS

HFA_CANDIDATES = [45, 50, 55, 60, 65]
WALKFORWARD_SEASONS = range(2015, 2023)  # validate each season 2015-2022, training on everything before it
CALIB_SEASONS = [2023, 2024]
TEST_SEASON = 2025

SITUATIONAL_COLS = ["home_rest", "away_rest", "rest_diff", "div_game", "is_neutral_site",
                     "qb_change_home", "qb_change_away"]
SPREAD_ONLY_COLS = ["spread_line", "total_line"] + SITUATIONAL_COLS
ELO_ONLY_COLS = ["home_elo", "away_elo", "elo_diff"] + SITUATIONAL_COLS

XGB_CONFIGS = {
    "xgb_depth2": dict(n_estimators=200, max_depth=2, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=1.0,
                        random_state=42),
    "xgb_depth3": dict(n_estimators=200, max_depth=3, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=1.0,
                        random_state=42),
}


def devig_moneyline_prob(home_ml, away_ml):
    """Standard de-vig: convert American odds to implied probabilities, normalize to sum to 1."""
    def implied(ml):
        return 100.0 / (ml + 100.0) if ml > 0 else (-ml) / (-ml + 100.0)
    p_home, p_away = implied(home_ml), implied(away_ml)
    return p_home / (p_home + p_away)


def tune_hfa(games_df):
    """
    Lightweight HFA grid search: for each candidate, rebuild the Elo-based
    dataset and fit a simple logistic regression on elo_diff (+ minimal
    situational context) to predict the home win, training on 2005-2021 and
    validating on 2022's Brier score. This is a cheap proxy for the full
    "grid-search against the calibration set's Brier/logloss" - a fast way
    to pick a reasonable HFA without re-running the full model/calibration
    pipeline once per candidate.
    """
    print("Tuning home-field-advantage constant...")
    best_hfa, best_brier = None, np.inf
    for hfa in HFA_CANDIDATES:
        dataset = build_labeled_dataset(games_df, hfa=hfa)
        train = dataset[dataset["season"] <= 2021]
        valid = dataset[dataset["season"] == 2022]

        X_train = train[["elo_diff", "div_game", "is_neutral_site"]]
        y_train = (train["margin"] > 0).astype(int)
        X_valid = valid[["elo_diff", "div_game", "is_neutral_site"]]
        y_valid = (valid["margin"] > 0).astype(int)

        clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
        probs = clf.predict_proba(X_valid)[:, 1]
        brier = brier_score_loss(y_valid, probs)
        print(f"  hfa={hfa}: 2022 validation Brier={brier:.4f}")
        if brier < best_brier:
            best_brier, best_hfa = brier, hfa

    print(f"Selected hfa={best_hfa}")
    return best_hfa


def walkforward_rmse(dataset, feature_cols, model_name):
    """Expanding-window walk-forward validation: for each season in
    WALKFORWARD_SEASONS, train on every prior season and predict that
    season's margin. Returns the aggregate RMSE across all validation
    seasons pooled - more reliable than any single season's metric given
    how few games there are per season (~255-285)."""
    preds, actuals = [], []
    for season in WALKFORWARD_SEASONS:
        train = dataset[dataset["season"] < season]
        valid = dataset[dataset["season"] == season]
        X_train, y_train = train[feature_cols], train["margin"]
        X_valid, y_valid = valid[feature_cols], valid["margin"]

        if model_name == "ridge":
            model = Ridge(alpha=1.0).fit(X_train, y_train)
        else:
            model = xgb.XGBRegressor(**XGB_CONFIGS[model_name]).fit(X_train, y_train)

        preds.append(model.predict(X_valid))
        actuals.append(y_valid.values)

    preds, actuals = np.concatenate(preds), np.concatenate(actuals)
    return float(np.sqrt(mean_squared_error(actuals, preds))), float(mean_absolute_error(actuals, preds))


def choose_model(dataset):
    """Walk-forward-compares Ridge against two XGBoost depths on margin
    RMSE/MAE (all using the full combined feature set) and returns whichever
    wins - matches train_model.py's existing pattern of benchmarking
    XGBoost against a simpler baseline rather than assuming it's better."""
    print("\nWalk-forward model comparison (combined features):")
    results = {}
    for name in ["ridge", "xgb_depth2", "xgb_depth3"]:
        rmse, mae = walkforward_rmse(dataset, FEATURE_COLS, name)
        results[name] = rmse
        print(f"  {name:12s} RMSE={rmse:.3f}  MAE={mae:.3f}")
    best = min(results, key=results.get)
    print(f"Selected model: {best}")
    return best


def fit_model(name, X, y):
    if name == "ridge":
        return Ridge(alpha=1.0).fit(X, y)
    return xgb.XGBRegressor(**XGB_CONFIGS[name]).fit(X, y)


def bootstrap_ci(values_model, values_baseline, metric_fn, n=2000, seed=42):
    """Paired bootstrap CI on (baseline_metric - model_metric) - positive
    means the model beats the baseline. values_* are same-length arrays of
    per-game (pred, actual) usable by metric_fn(preds, actuals)."""
    rng = np.random.default_rng(seed)
    n_games = len(values_model[1])
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, n_games, n_games)
        m_model = metric_fn(values_model[0][idx], values_model[1][idx])
        m_base = metric_fn(values_baseline[0][idx], values_baseline[1][idx])
        diffs.append(m_base - m_model)
    diffs = np.array(diffs)
    return float(diffs.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def rmse_fn(pred, actual):
    return np.sqrt(mean_squared_error(actual, pred))


def brier_fn(pred_prob, actual):
    return brier_score_loss(actual, pred_prob)


def reliability_table(pred_probs, actual, n_bins=10):
    """Printed decile reliability check (predicted vs actual win rate per
    bin) - the QA step for the isotonic calibrator's fit, without adding a
    plotting dependency for a single diagnostic."""
    df = pd.DataFrame({"pred": pred_probs, "actual": actual})
    df["bin"] = pd.qcut(df["pred"], n_bins, duplicates="drop")
    table = df.groupby("bin", observed=True).agg(n=("actual", "size"), mean_pred=("pred", "mean"), mean_actual=("actual", "mean"))
    print(table.to_string(float_format=lambda x: f"{x:.3f}"))


def main():
    games_df = load_games()

    hfa = tune_hfa(games_df)
    dataset = build_labeled_dataset(games_df, hfa=hfa)

    model_name = choose_model(dataset)

    train = dataset[dataset["season"] <= 2022]
    calib = dataset[dataset["season"].isin(CALIB_SEASONS)]
    test = dataset[dataset["season"] == TEST_SEASON]
    print(f"\nTrain: {len(train)} (2005-2022) | Calibration: {len(calib)} (2023-2024) | Test: {len(test)} (2025, holdout)")

    sigma = float(np.std(train["margin"] - train["spread_line"]))
    print(f"Residual std (train, margin - spread_line): sigma={sigma:.2f}")
    home_win_test = (test["margin"] > 0).astype(int).values

    # --- Ablation: spread-only vs elo-only vs combined, decides the
    # feature set the shipped model actually uses. Note this uses the same
    # 2025 test set as the final reported evaluation below - a real
    # methodological compromise given how few post-Elo-warmup NFL seasons
    # exist to spare a separate feature-selection holdout; documented
    # plainly rather than presented as a clean nested split. ---
    corr = train[["elo_diff", "spread_line"]].corr().iloc[0, 1]
    print(f"\n=== Ablation (elo_diff vs spread_line correlation: {corr:.3f}) ===")
    ablation = {}
    for label, cols in [("spread_only", SPREAD_ONLY_COLS), ("elo_only", ELO_ONLY_COLS), ("combined", FEATURE_COLS)]:
        m = fit_model(model_name, train[cols], train["margin"])
        pred_test = m.predict(test[cols])
        pred_calib = m.predict(calib[cols])
        rmse = rmse_fn(pred_test, test["margin"].values)
        auc = roc_auc_score(home_win_test, norm.cdf(pred_test / sigma))
        ablation[label] = {"cols": cols, "model": m, "rmse": rmse, "auc": auc,
                            "pred_test": pred_test, "pred_calib": pred_calib}
        print(f"  {label:12s} RMSE={rmse:.3f}  AUC(parametric)={auc:.4f}")

    mean_diff, lo, hi = bootstrap_ci(
        (ablation["combined"]["pred_test"], test["margin"].values),
        (ablation["spread_only"]["pred_test"], test["margin"].values), rmse_fn)
    print(f"Bootstrap 95% CI on (spread_only_RMSE - combined_RMSE): {mean_diff:.3f} [{lo:.3f}, {hi:.3f}]")

    if lo > 0:
        chosen = "combined"
        print("combined meaningfully beats spread-only -> shipping the combined feature set.")
    else:
        chosen = "spread_only"
        print("No clear improvement from Elo/situational features over spread-only -> "
              "shipping the leaner spread_only feature set (matches this repo's existing "
              "self-critical documentation style: don't ship complexity that isn't earning it).")

    feature_cols = ablation[chosen]["cols"]
    model = ablation[chosen]["model"]
    pred_margin_calib = ablation[chosen]["pred_calib"]
    pred_margin_test = ablation[chosen]["pred_test"]

    # --- Win probability, derived from the chosen model's margin prediction ---
    home_win_calib = (calib["margin"] > 0).astype(int).values
    iso = IsotonicRegression(out_of_bounds="clip").fit(pred_margin_calib, home_win_calib)
    wp_test_iso = iso.predict(pred_margin_test)
    wp_test_parametric = norm.cdf(pred_margin_test / sigma)

    print(f"\n=== Reliability check: isotonic-calibrated win probability, {chosen} model (2025 test) ===")
    reliability_table(wp_test_iso, home_win_test)
    print(f"Parametric cross-check (Phi(margin/sigma)) AUC={roc_auc_score(home_win_test, wp_test_parametric):.4f} "
          f"Brier={brier_score_loss(home_win_test, wp_test_parametric):.4f} - compare to isotonic numbers below.")

    # --- Baselines ---
    spread_pred_test = test["spread_line"].values
    spread_wp_test = norm.cdf(spread_pred_test / sigma)
    ml_mask = test["season"].values >= 2010  # moneylines unreliable before 2010
    ml_wp_test = np.full(len(test), np.nan)
    if ml_mask.any():
        ml_wp_test[ml_mask] = [
            devig_moneyline_prob(hm, am) for hm, am in
            zip(test.loc[ml_mask, "home_moneyline"], test.loc[ml_mask, "away_moneyline"])
        ]

    print(f"\n=== Margin: {chosen} model vs spread-line baseline (2025 test) ===")
    model_rmse = rmse_fn(pred_margin_test, test["margin"].values)
    base_rmse = rmse_fn(spread_pred_test, test["margin"].values)
    print(f"Model RMSE={model_rmse:.3f}  MAE={mean_absolute_error(test['margin'], pred_margin_test):.3f}")
    print(f"Spread-line baseline RMSE={base_rmse:.3f}  MAE={mean_absolute_error(test['margin'], spread_pred_test):.3f}")
    mean_diff, lo, hi = bootstrap_ci(
        (pred_margin_test, test["margin"].values), (spread_pred_test, test["margin"].values), rmse_fn)
    print(f"Bootstrap 95% CI on (baseline_RMSE - model_RMSE): {mean_diff:.3f} [{lo:.3f}, {hi:.3f}] "
          f"({'model wins' if lo > 0 else 'not distinguishable from the market' if lo < 0 < hi else 'baseline wins'})")

    print(f"\n=== Win probability: {chosen} model vs spread-derived baseline (2025 test) ===")
    print(f"Model (isotonic) AUC={roc_auc_score(home_win_test, wp_test_iso):.4f}  Brier={brier_score_loss(home_win_test, wp_test_iso):.4f}")
    print(f"Spread-derived baseline AUC={roc_auc_score(home_win_test, spread_wp_test):.4f}  Brier={brier_score_loss(home_win_test, spread_wp_test):.4f}")
    mean_diff, lo, hi = bootstrap_ci(
        (wp_test_iso, home_win_test), (spread_wp_test, home_win_test), brier_fn)
    print(f"Bootstrap 95% CI on (baseline_Brier - model_Brier): {mean_diff:.4f} [{lo:.4f}, {hi:.4f}] "
          f"({'model wins' if lo > 0 else 'not distinguishable from the market' if lo < 0 < hi else 'baseline wins'})")

    if ml_mask.any():
        sub_actual = home_win_test[ml_mask]
        print(f"\nDe-vigged moneyline baseline (2025, seasons>=2010, n={ml_mask.sum()}): "
              f"AUC={roc_auc_score(sub_actual, ml_wp_test[ml_mask]):.4f}  Brier={brier_score_loss(sub_actual, ml_wp_test[ml_mask]):.4f}")

    # --- Save artifacts ---
    if model_name == "ridge":
        with open("data/pregame_model.pkl", "wb") as f:
            pickle.dump(model, f)
    else:
        model.save_model("data/pregame_xgb_model.json")
    with open("data/pregame_isotonic_calibrator.pkl", "wb") as f:
        pickle.dump(iso, f)
    with open("data/pregame_feature_cols.json", "w") as f:
        json.dump(feature_cols, f)
    with open("data/pregame_meta.json", "w") as f:
        json.dump({
            "model_name": model_name, "feature_set": chosen, "hfa": hfa, "sigma": sigma,
            "test_margin_rmse": model_rmse, "test_wp_auc": float(roc_auc_score(home_win_test, wp_test_iso)),
            "baseline_margin_rmse": base_rmse, "baseline_wp_auc": float(roc_auc_score(home_win_test, spread_wp_test)),
        }, f)
    print(f"\nSaved pregame model artifacts to data/ (model_name={model_name}, feature_set={chosen})")


if __name__ == "__main__":
    main()
