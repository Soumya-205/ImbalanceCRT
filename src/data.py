"""
Data loading and imbalance-sweep construction for ImbalanceCRT.

Downloads the ULB Credit Card Fraud dataset, builds stratified
train/calibration/test splits, and subsamples the majority class
to produce controlled positive-class ratios for the imbalance sweep.
"""

import numpy as np
import pandas as pd
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import DATA_DIR, IMBALANCE_RATIOS, TRAIN_FRAC, CAL_FRAC, TEST_FRAC


def download_fraud_dataset() -> pd.DataFrame:
    """Download (if needed) and load the ULB Credit Card Fraud dataset."""
    path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    csv_path = f"{path}/creditcard.csv"
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows, {df['Class'].mean():.4%} positive (fraud) rate")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Scale Time and Amount; V1-V28 are already PCA'd and pre-scaled."""
    df = df.copy()
    scaler = StandardScaler()
    df[["Time", "Amount"]] = scaler.fit_transform(df[["Time", "Amount"]])
    return df


def make_imbalance_level(df: pd.DataFrame, target_ratio: float, seed: int) -> pd.DataFrame:
    """
    Subsample the majority (legitimate) class so the positive (fraud)
    class makes up `target_ratio` of the resulting dataset.
    All minority-class rows are kept; only the majority class is downsampled,
    which isolates imbalance as the sole variable across the sweep.
    """
    pos = df[df["Class"] == 1]
    neg = df[df["Class"] == 0]

    n_pos = len(pos)
    n_neg_needed = int(n_pos * (1 - target_ratio) / target_ratio)
    n_neg_needed = min(n_neg_needed, len(neg))  # can't exceed available negatives

    neg_sampled = neg.sample(n=n_neg_needed, random_state=seed)
    out = pd.concat([pos, neg_sampled]).sample(frac=1, random_state=seed)  # shuffle
    print(f"  ratio={target_ratio:.0%} -> {len(out)} rows "
          f"({n_pos} pos / {n_neg_needed} neg, actual={n_pos/len(out):.2%})")
    return out.reset_index(drop=True)


def split_train_cal_test(df: pd.DataFrame, seed: int):
    """Stratified train/calibration/test split, preserving class ratio in each."""
    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, train_size=TRAIN_FRAC, stratify=y, random_state=seed
    )
    rel_cal = CAL_FRAC / (CAL_FRAC + TEST_FRAC)
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_rest, y_rest, train_size=rel_cal, stratify=y_rest, random_state=seed
    )
    return (X_train, y_train), (X_cal, y_cal), (X_test, y_test)


def get_imbalance_sweep_splits(seed: int = 0):
    """
    Full pipeline: download, preprocess, and produce train/cal/test splits
    for every imbalance level in IMBALANCE_RATIOS, at a given seed.
    Returns: dict mapping ratio -> (train, cal, test)
    """
    df = preprocess(download_fraud_dataset())
    sweep = {}
    print(f"\nBuilding imbalance sweep at seed={seed}:")
    for ratio in IMBALANCE_RATIOS:
        level_df = make_imbalance_level(df, ratio, seed=seed)
        sweep[ratio] = split_train_cal_test(level_df, seed=seed)
    return sweep


if __name__ == "__main__":
    sweep = get_imbalance_sweep_splits(seed=0)
    for ratio, (train, cal, test) in sweep.items():
        (X_tr, y_tr), (X_cal, y_cal), (X_te, y_te) = (train, cal, test)
        print(f"ratio={ratio}: train={len(X_tr)}, cal={len(X_cal)}, test={len(X_te)}")