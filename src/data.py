"""
Data loading and imbalance-sweep construction for ImbalanceCRT.

Two datasets share one pipeline:
  * "fraud"        ULB Credit Card Fraud (kagglehub mlg-ulb/creditcardfraud).
                   All 492 frauds kept at every ratio.
  * "lending_club" Lending Club accepted loans 2007-2018 (kagglehub
                   wordsforthewise/lending-club). Resolved loans only
                   (Fully Paid vs Charged Off / Default); LC_N_POS defaults
                   sampled per seed and kept at every ratio; only information
                   known when the loan was issued (no leakage columns).

For each ratio: keep the positives, subsample negatives to hit the ratio, make a
stratified train/cal/test split, then impute missing values and standardise
using statistics from the TRAINING split only.
"""
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import DATA_DIR, IMBALANCE_RATIOS, TRAIN_FRAC, CAL_FRAC, TEST_FRAC, LC_N_POS, BAF_N_POS


# ===================================================================== fraud
FRAUD_SCALE_COLS = ["Time", "Amount"]  # V1-V28 are PCA outputs already


@lru_cache(maxsize=1)
def load_fraud_dataset() -> pd.DataFrame:
    """Download (first time only) and read the CSV. Cached: runs once per process."""
    path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    df = pd.read_csv(f"{path}/creditcard.csv")
    print(f"Loaded {len(df)} rows, {df['Class'].mean():.4%} positive rate, "
          f"{df.duplicated().sum()} exact duplicate rows")
    return df


# ============================================================== lending club
# Raw columns read from the CSV. Every one is known at the moment the loan is
# issued. Columns describing what happened afterwards (payments, recoveries,
# last FICO, settlement, hardship, ...) are never read, so they cannot leak.
LC_RAW_COLS = [
    "loan_status", "issue_d",                                   # label; date (only for credit age)
    "loan_amnt", "term", "int_rate", "installment", "grade",
    "emp_length", "home_ownership", "annual_inc", "verification_status",
    "purpose", "dti", "delinq_2yrs", "earliest_cr_line",
    "fico_range_low", "fico_range_high", "inq_last_6mths", "open_acc",
    "pub_rec", "revol_bal", "revol_util", "total_acc", "mort_acc",
    "pub_rec_bankruptcies", "application_type", "initial_list_status",
]
LC_STATUS = {"Fully Paid": 0, "Charged Off": 1, "Default": 1}   # resolved loans only
LC_NUM_COLS = [                                                  # imputed + standardised
    "loan_amnt", "term", "int_rate", "installment", "emp_length", "log_annual_inc",
    "dti", "delinq_2yrs", "fico", "inq_last_6mths", "open_acc", "pub_rec",
    "log_revol_bal", "revol_util", "total_acc", "mort_acc", "pub_rec_bankruptcies",
    "credit_age_months",
]
LC_CAT_COLS = ["grade", "home_ownership", "verification_status", "purpose",
               "application_type", "initial_list_status"]       # one-hot encoded
LC_CACHE = DATA_DIR / "lending_club_clean.pkl"


def _find_lc_csv(root: str) -> Path:
    """The Kaggle download contains accepted_... and rejected_... files; find the accepted one."""
    hits = [p for p in Path(root).rglob("accepted*") if p.is_file()
            and (p.name.endswith(".csv") or p.name.endswith(".csv.gz"))]
    if not hits:
        raise FileNotFoundError(f"No accepted_*.csv(.gz) under {root}. Contents: "
                                f"{[p.name for p in Path(root).rglob('*')][:20]}")
    return sorted(hits, key=lambda p: p.stat().st_size)[-1]     # largest match


def _percent(s: pd.Series) -> pd.Series:
    """'13.56%' or 13.56 -> 13.56 (handles both formats)."""
    return pd.to_numeric(s.astype(str).str.replace("%", "", regex=False).str.strip(),
                         errors="coerce")


def _emp_length(s: pd.Series) -> pd.Series:
    """'< 1 year' -> 0, '1 year' -> 1, '10+ years' -> 10, missing / 'n/a' -> NaN."""
    s = s.astype(str).str.strip()
    out = pd.to_numeric(s.str.extract(r"(\d+)")[0], errors="coerce")
    out[s.str.startswith("<")] = 0
    return out


def clean_lending_club(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw accepted-loans table -> model-ready table with a 0/1 'Class' column."""
    df = raw[raw["loan_status"].isin(LC_STATUS)].copy()
    out = pd.DataFrame(index=df.index)
    out["Class"] = df["loan_status"].map(LC_STATUS).astype(int)

    out["loan_amnt"] = pd.to_numeric(df["loan_amnt"], errors="coerce")
    out["term"] = pd.to_numeric(df["term"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    out["int_rate"] = _percent(df["int_rate"])
    out["installment"] = pd.to_numeric(df["installment"], errors="coerce")
    out["emp_length"] = _emp_length(df["emp_length"])
    out["log_annual_inc"] = np.log1p(pd.to_numeric(df["annual_inc"], errors="coerce").clip(lower=0))
    out["dti"] = pd.to_numeric(df["dti"], errors="coerce")
    out["delinq_2yrs"] = pd.to_numeric(df["delinq_2yrs"], errors="coerce")
    out["fico"] = (pd.to_numeric(df["fico_range_low"], errors="coerce")
                   + pd.to_numeric(df["fico_range_high"], errors="coerce")) / 2
    for c in ["inq_last_6mths", "open_acc", "pub_rec", "total_acc", "mort_acc",
              "pub_rec_bankruptcies"]:
        out[c] = pd.to_numeric(df[c], errors="coerce")
    out["log_revol_bal"] = np.log1p(pd.to_numeric(df["revol_bal"], errors="coerce").clip(lower=0))
    out["revol_util"] = _percent(df["revol_util"])

    # Length of credit history when the loan was issued, in months.
    issue = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    first = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce")
    out["credit_age_months"] = ((issue.dt.year - first.dt.year) * 12
                                + (issue.dt.month - first.dt.month))

    cats = df[LC_CAT_COLS].astype(str).apply(lambda s: s.str.strip())
    cats["home_ownership"] = cats["home_ownership"].where(
        cats["home_ownership"].isin(["MORTGAGE", "RENT", "OWN"]), "OTHER")  # ANY/NONE/OTHER are tiny
    dummies = pd.get_dummies(cats, prefix=LC_CAT_COLS, dtype=np.float32)

    out = pd.concat([out, dummies], axis=1)
    feat = out.columns.drop("Class")
    out[feat] = out[feat].astype(np.float32)
    return out.reset_index(drop=True)


@lru_cache(maxsize=1)
def load_lending_club() -> pd.DataFrame:
    """Download (first time only), clean, and cache to data/lending_club_clean.pkl.
    Later runs read the small cached file instead of the ~1.6 GB CSV."""
    if LC_CACHE.exists():
        df = pd.read_pickle(LC_CACHE)
    else:
        csv = _find_lc_csv(kagglehub.dataset_download("wordsforthewise/lending-club"))
        print(f"Reading {csv.name} (first run only; takes a few minutes)...")
        raw = pd.read_csv(csv, usecols=LC_RAW_COLS, low_memory=False)
        df = clean_lending_club(raw)
        df.to_pickle(LC_CACHE)
        print(f"Cached cleaned table to {LC_CACHE}")
    print(f"Loaded {len(df)} resolved loans, {df['Class'].mean():.2%} default rate, "
          f"{df.shape[1] - 1} features, {int(df.isna().any(axis=1).sum())} rows with a missing value")
    return df

# ======================================================= bank account fraud (BAF)
# Jesus et al., NeurIPS 2022 Datasets & Benchmarks: synthetic data generated from a
# real bank account-opening fraud dataset. Base variant only (Base.csv).
BAF_LABEL = "fraud_bool"
BAF_DROP = ["month"]                       # time index; splits here are random
BAF_MISSING_NEG1 = ["prev_address_months_count", "current_address_months_count",
                    "bank_months_count", "session_length_in_minutes",
                    "device_distinct_emails_8w"]           # -1 means missing
BAF_MISSING_NEG = ["intended_balcon_amount"]               # any negative value means missing


@lru_cache(maxsize=1)
def load_baf() -> pd.DataFrame:
    """Download (first time only) and clean BAF Base.csv -> numeric table with 'Class'.
    Categorical (text) columns are one-hot encoded; missing codes become NaN and are
    filled later with the TRAIN median. Sets DATASETS['baf']['scale_cols']."""
    csv = Path(kagglehub.dataset_download("sgpjesus/bank-account-fraud-dataset-neurips-2022",
                                          path="Base.csv"))      # only the base file
    if zipfile.is_zipfile(csv):                                  # Kaggle may deliver it zipped
        with zipfile.ZipFile(csv) as z:
            name = next(n for n in z.namelist() if n.endswith(".csv"))
            raw = pd.read_csv(z.open(name))
    else:
        raw = pd.read_csv(csv)
    df = raw.drop(columns=[c for c in BAF_DROP if c in raw.columns])
    for c in BAF_MISSING_NEG1:
        if c in df.columns:
            df.loc[df[c] == -1, c] = np.nan
    for c in BAF_MISSING_NEG:
        if c in df.columns:
            df.loc[df[c] < 0, c] = np.nan
    y = df.pop(BAF_LABEL).astype(int)
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]   # text columns
    num_cols = [c for c in df.columns if c not in cat_cols]
    df = pd.concat([df[num_cols].astype(np.float32),
                    pd.get_dummies(df[cat_cols], prefix=cat_cols, dtype=np.float32)], axis=1)
    df["Class"] = y.values
    DATASETS["baf"]["scale_cols"] = num_cols
    print(f"Loaded {len(df)} BAF applications, {df['Class'].mean():.2%} fraud rate, "
          f"{df.shape[1] - 1} features ({len(num_cols)} numeric, {len(cat_cols)} categorical -> one-hot)")
    return df


# ================================================================ shared pipeline
DATASETS = {
    "fraud": dict(load=load_fraud_dataset, n_pos=None, scale_cols=FRAUD_SCALE_COLS),
    "lending_club": dict(load=load_lending_club, n_pos=LC_N_POS, scale_cols=LC_NUM_COLS),
    "baf": dict(load=load_baf, n_pos=BAF_N_POS, scale_cols=None),   # scale_cols set by load_baf
}


def make_imbalance_level(pos: pd.DataFrame, neg: pd.DataFrame,
                         target_ratio: float, seed: int) -> pd.DataFrame:
    """Keep every given positive; subsample negatives so positives = target_ratio."""
    n_neg = min(int(len(pos) * (1 - target_ratio) / target_ratio), len(neg))
    out = pd.concat([pos, neg.sample(n=n_neg, random_state=seed)])
    out = out.sample(frac=1, random_state=seed).reset_index(drop=True)
    print(f"  ratio={target_ratio:.2%} -> {len(out)} rows "
          f"({len(pos)} pos / {n_neg} neg, actual={len(pos)/len(out):.2%})")
    return out


def split_and_scale(df: pd.DataFrame, seed: int, scale_cols):
    """Stratified train/cal/test split; then fill missing values with the TRAIN
    median and standardise with TRAIN mean/std (no statistics from cal/test)."""
    X, y = df.drop(columns=["Class"]), df["Class"]
    X_tr, X_rest, y_tr, y_rest = train_test_split(
        X, y, train_size=TRAIN_FRAC, stratify=y, random_state=seed)
    X_cal, X_te, y_cal, y_te = train_test_split(
        X_rest, y_rest, train_size=CAL_FRAC / (CAL_FRAC + TEST_FRAC),
        stratify=y_rest, random_state=seed)

    X_tr, X_cal, X_te = X_tr.copy(), X_cal.copy(), X_te.copy()
    medians = X_tr[scale_cols].median()                          # learned from train only
    scaler = StandardScaler().fit(X_tr[scale_cols].fillna(medians))
    for part in (X_tr, X_cal, X_te):
        part[scale_cols] = scaler.transform(part[scale_cols].fillna(medians))
    return (X_tr, y_tr), (X_cal, y_cal), (X_te, y_te)


def get_imbalance_sweep_splits(seed: int = 0, ratios=IMBALANCE_RATIOS, dataset: str = "fraud"):
    """Returns dict: ratio -> ((X_tr, y_tr), (X_cal, y_cal), (X_te, y_te))."""
    spec = DATASETS[dataset]
    df = spec["load"]()
    pos, neg = df[df["Class"] == 1], df[df["Class"] == 0]
    if spec["n_pos"] is not None:                                # same positives at every ratio
        pos = pos.sample(n=spec["n_pos"], random_state=seed)
    print(f"\nBuilding {dataset} imbalance sweep at seed={seed}:")
    return {r: split_and_scale(make_imbalance_level(pos, neg, r, seed), seed, spec["scale_cols"])
            for r in ratios}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=list(DATASETS), default="fraud")
    name = p.parse_args().dataset
    for ratio, ((X_tr, y_tr), (X_cal, y_cal), (X_te, y_te)) in get_imbalance_sweep_splits(0, dataset=name).items():
        print(f"ratio={ratio}: train={len(X_tr)} ({int(y_tr.sum())} pos), "
              f"cal={len(X_cal)} ({int(y_cal.sum())} pos), test={len(X_te)} ({int(y_te.sum())} pos), "
              f"features={X_tr.shape[1]}, NaNs after imputation={int(X_tr.isna().sum().sum())}")