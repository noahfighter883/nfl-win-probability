"""
Win Probability Model - Training
Train on 2021-2023, hold out 2024 entirely as the test set.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.isotonic import IsotonicRegression
import xgboost as xgb
import json

FEATURE_COLS = [
    "score_differential", "game_seconds_remaining", "yardline_100", "ydstogo",
    "posteam_timeouts_remaining", "defteam_timeouts_remaining", "score_diff_x_time",
    "is_home", "is_two_minute_drill", "is_garbage_time", "is_red_zone",
    "down_1", "down_2", "down_3", "down_4"
]

def main():
    df = pd.read_parquet("data/model_dataset.parquet")

    # Three-way season split:
    #   2021-2022 -> fit the model
    #   2023      -> fit isotonic calibration (never seen by the model during training)
    #   2024      -> final untouched holdout for reporting metrics
    train = df[df["season"].isin([2021, 2022])]
    calib = df[df["season"] == 2023]
    test = df[df["season"] == 2024]
    print(f"Train: {len(train)} (2021-2022) | Calibration: {len(calib)} (2023) | Test: {len(test)} (2024, holdout)")

    X_train, y_train = train[FEATURE_COLS], train["label"]
    X_calib, y_calib = calib[FEATURE_COLS], calib["label"]
    X_test, y_test = test[FEATURE_COLS], test["label"]

    # --- Baseline: Logistic Regression ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train_scaled, y_train)
    logreg_probs = logreg.predict_proba(X_test_scaled)[:, 1]

    print("\n=== Logistic Regression (uncalibrated, for reference) ===")
    print(f"AUC: {roc_auc_score(y_test, logreg_probs):.4f}")
    print(f"Brier score: {brier_score_loss(y_test, logreg_probs):.4f}")

    # --- XGBoost ---
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    )
    xgb_model.fit(X_train, y_train)
    xgb_probs_raw_test = xgb_model.predict_proba(X_test)[:, 1]

    print("\n=== XGBoost (uncalibrated) ===")
    print(f"AUC: {roc_auc_score(y_test, xgb_probs_raw_test):.4f}")
    print(f"Brier score: {brier_score_loss(y_test, xgb_probs_raw_test):.4f}")

    # --- Isotonic calibration, fit on the 2023 calibration set ---
    xgb_probs_raw_calib = xgb_model.predict_proba(X_calib)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(xgb_probs_raw_calib, y_calib)

    xgb_probs_calibrated = iso.predict(xgb_probs_raw_test)

    print("\n=== XGBoost (isotonic calibrated) ===")
    print(f"AUC: {roc_auc_score(y_test, xgb_probs_calibrated):.4f}")
    print(f"Brier score: {brier_score_loss(y_test, xgb_probs_calibrated):.4f}")

    # Feature importance
    importance = dict(zip(FEATURE_COLS, xgb_model.feature_importances_.tolist()))
    importance = dict(sorted(importance.items(), key=lambda x: -x[1]))
    print("\nFeature importances (XGBoost):")
    for k, v in importance.items():
        print(f"  {k}: {v:.4f}")

    # Save test set predictions for calibration analysis + frontend
    test_out = test[["game_id", "season", "posteam", "defteam", "home_team",
                      "away_team", "qtr", "desc", "label"]].copy()
    test_out["logreg_prob"] = logreg_probs
    test_out["xgb_prob_raw"] = xgb_probs_raw_test
    test_out["xgb_prob"] = xgb_probs_calibrated  # calibrated becomes the "main" prediction
    test_out.to_parquet("data/test_predictions.parquet", index=False)

    # Save models info
    xgb_model.save_model("data/xgb_model.json")
    with open("data/feature_cols.json", "w") as f:
        json.dump(FEATURE_COLS, f)
    import pickle
    with open("data/isotonic_calibrator.pkl", "wb") as f:
        pickle.dump(iso, f)

    print("\nSaved test predictions, model, and calibrator to data/")

if __name__ == "__main__":
    main()
