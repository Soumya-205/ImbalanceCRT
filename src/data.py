"""
Data loading and imbalance-sweep construction for ImbalanceCRT.

Loads the ULB Credit Card Fraud dataset once, subsamples the majority
class to hit each target positive ratio, makes stratified
train/calibration/test splits, and fits the scaler on TRAIN ONLY.
"""

from functools import lru_cache

import pandas as pd
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import IMBALANCE_RATIOS, TRAIN_FRAC, CAL_FRAC, TEST_FRAC

SCALE_COLS = ["Time", "Amount"]  # V1-V28 are PCA outputs already


@lru_cache(maxsize=1)
def load_fraud_dataset() -> pd.DataFrame:
    """Download (first time only) and read the CSV. Cached: runs once per process."""
    path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    df = pd.read_csv(f"{path}/creditcard.csv")
    print(f"Loaded {len(df)} rows, {df['Class'].mean():.4%} positive rate, "
          f"{df.duplicated().sum()} exact duplicate rows")
    return df


def make_imbalance_level(df: pd.DataFrame, target_ratio: float, seed: int) -> pd.DataFrame:
    """Keep every positive; subsample negatives so positives = target_ratio."""
    pos = df[df["Class"] == 1]
    neg = df[df["Class"] == 0]
    n_neg = min(int(len(pos) * (1 - target_ratio) / target_ratio), len(neg))
    out = pd.concat([pos, neg.sample(n=n_neg, random_state=seed)])
    out = out.sample(frac=1, random_state=seed).reset_index(drop=True)
    print(f"  ratio={target_ratio:.2%} -> {len(out)} rows "
          f"({len(pos)} pos / {n_neg} neg, actual={len(pos)/len(out):.2%})")
    return out


def split_and_scale(df: pd.DataFrame, seed: int):
    """Stratified train/cal/test split, then scale using TRAIN statistics only."""
    X, y = df.drop(columns=["Class"]), df["Class"]
    X_tr, X_rest, y_tr, y_rest = train_test_split(
        X, y, train_size=TRAIN_FRAC, stratify=y, random_state=seed)
    X_cal, X_te, y_cal, y_te = train_test_split(
        X_rest, y_rest, train_size=CAL_FRAC / (CAL_FRAC + TEST_FRAC),
        stratify=y_rest, random_state=seed)

    X_tr, X_cal, X_te = X_tr.copy(), X_cal.copy(), X_te.copy()
    scaler = StandardScaler().fit(X_tr[SCALE_COLS])            # learn mean/std from train
    for part in (X_tr, X_cal, X_te):
        part[SCALE_COLS] = scaler.transform(part[SCALE_COLS])  # apply to all three
    return (X_tr, y_tr), (X_cal, y_cal), (X_te, y_te)


def get_imbalance_sweep_splits(seed: int = 0, ratios=IMBALANCE_RATIOS):
    """Returns dict: ratio -> ((X_tr, y_tr), (X_cal, y_cal), (X_te, y_te))."""
    df = load_fraud_dataset()
    print(f"\nBuilding imbalance sweep at seed={seed}:")
    return {r: split_and_scale(make_imbalance_level(df, r, seed), seed) for r in ratios}


if __name__ == "__main__":
    for ratio, ((X_tr, y_tr), (X_cal, y_cal), (X_te, y_te)) in get_imbalance_sweep_splits(0).items():
        print(f"ratio={ratio}: train={len(X_tr)} ({int(y_tr.sum())} pos), "
              f"cal={len(X_cal)} ({int(y_cal.sum())} pos), test={len(X_te)} ({int(y_te.sum())} pos)")