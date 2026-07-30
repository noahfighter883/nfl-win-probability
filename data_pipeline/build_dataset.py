"""
Win Probability Model - Dataset builder
Loads nflverse play-by-play data, constructs the win/loss label,
and engineers features for modeling.
"""
import pandas as pd
import numpy as np
import os

from features import build_label, engineer_features

YEARS = [2021, 2022, 2023, 2024]

def download_season(year):
    """Downloads pbp data from nflverse-data GitHub releases if not already present."""
    os.makedirs("data", exist_ok=True)
    path = f"data/pbp_{year}.csv.gz"
    if not os.path.exists(path):
        url = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{year}.csv.gz"
        print(f"Downloading {year} play-by-play data...")
        import urllib.request
        urllib.request.urlretrieve(url, path)
    return path

def load_season(year):
    path = download_season(year)
    df = pd.read_csv(path, compression="gzip", low_memory=False)
    df["season"] = year
    return df

def main():
    print("Loading seasons...")
    all_dfs = [load_season(y) for y in YEARS]
    df = pd.concat(all_dfs, ignore_index=True)
    print(f"Loaded {len(df)} total plays across {len(YEARS)} seasons")

    df = build_label(df)
    print(f"After dropping ties: {len(df)} plays")

    features, meta = engineer_features(df)
    print(f"After feature engineering: {len(features)} plays")
    print(f"Feature columns: {list(features.columns)}")

    dataset = pd.concat([meta.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    dataset.to_parquet("data/model_dataset.parquet", index=False)
    print("Saved to data/model_dataset.parquet")
    print()
    print("Label balance:", dataset["label"].mean())
    print("Rows per season:")
    print(dataset.groupby("season").size())

if __name__ == "__main__":
    main()
