"""Models A (BCE) and B (class-weighted BCE), each followed by post-hoc CRC.

Usage (from the project root):
    python src/baselines.py --quick    # 1 seed, 5% ratio, prints only
    python src/baselines.py            # full sweep -> results/fraud_baselines.csv
"""
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from config import (RESULTS_DIR, RANDOM_SEEDS, IMBALANCE_RATIOS, TARGET_FNR_LEVELS,
                    HIDDEN, EPOCHS, LR, BATCH_SIZE, N_RESPLITS)
from models import MLP, predict_logits
from crc import crc_threshold_fnr, fnr_fpr, oracle_fpr
from data import get_imbalance_sweep_splits

MODELS = {"A": False, "B": True}   # model name -> use class weighting?


def to_np(X, y):
    return X.values.astype(np.float32), y.values.astype(np.float32)


def train_bce(X, y, class_weighted, seed):
    torch.manual_seed(seed)                      # same init for A and B at a given seed
    model = MLP(X.shape[1], hidden=HIDDEN)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    pw = None
    if class_weighted:
        n_pos = float(y.sum())
        pw = torch.tensor([(len(y) - n_pos) / n_pos])   # e.g. 99 at 1% positives
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    model.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(len(yt))
        for i in range(0, len(yt), BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            opt.zero_grad()
            loss_fn(model(Xt[idx]), yt[idx]).backward()
            opt.step()
    return model


def evaluate(model, cal, test, seed, n_resplits=N_RESPLITS):
    """Post-hoc CRC averaged over random calibration/test re-splits.
    The model is trained once. Its held-out data (cal + test) is pooled; each
    re-split draws a fresh calibration set with the original number of positives
    and negatives, fits the CRC threshold on it, and scores the remainder.
    Seeded by `seed`, so Models A, B and C see identical re-splits (paired)."""
    X = np.concatenate([cal[0], test[0]])
    y = np.concatenate([cal[1], test[1]])
    s = predict_logits(model, X)                       # score the pooled set once
    n_cal_pos = int(cal[1].sum())
    n_cal_neg = len(cal[1]) - n_cal_pos
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)

    acc = {a: [] for a in TARGET_FNR_LEVELS}
    for _ in range(n_resplits):
        p, n = rng.permutation(pos), rng.permutation(neg)
        cal_pos = p[:n_cal_pos]                                  # CRC needs only cal positives
        te = np.concatenate([p[n_cal_pos:], n[n_cal_neg:]])      # everything not in cal
        for a in TARGET_FNR_LEVELS:                              # same split for both alphas
            lam, ok = crc_threshold_fnr(s[cal_pos], a)
            fnr, fpr = fnr_fpr(y[te], s[te], lam)
            acc[a].append((fnr, fpr, oracle_fpr(y[te], s[te], a), ok))

    auroc = roc_auc_score(y, s)
    rows = []
    for a in TARGET_FNR_LEVELS:
        m = np.array(acc[a], dtype=float)
        rows.append(dict(alpha=a,
                         fnr=m[:, 0].mean(), fpr=m[:, 1].mean(), oracle_fpr=m[:, 2].mean(),
                         fnr_sd_split=m[:, 0].std(), fpr_sd_split=m[:, 1].std(),
                         feasible=bool(m[:, 3].all()), auroc=auroc, n_cal_pos=n_cal_pos))
    return rows


def main(quick):
    seeds = [0] if quick else RANDOM_SEEDS
    ratios = [0.05] if quick else IMBALANCE_RATIOS
    rows = []
    for seed in seeds:
        sweep = get_imbalance_sweep_splits(seed, ratios=ratios)
        for ratio, (train, cal, test) in sweep.items():
            train, cal, test = to_np(*train), to_np(*cal), to_np(*test)
            for name, cw in MODELS.items():
                model = train_bce(*train, class_weighted=cw, seed=seed)
                for r in evaluate(model, cal, test, seed):
                    rows.append(dict(dataset="fraud", model=name, seed=seed, ratio=ratio, **r))
                    print(f"seed={seed} ratio={ratio:.2f} model={name} alpha={r['alpha']:.2f} | "
                          f"FNR={r['fnr']:.3f} FPR={r['fpr']:.4f} "
                          f"oracleFPR={r['oracle_fpr']:.4f} AUROC={r['auroc']:.4f}")

    df = pd.DataFrame(rows)
    if not quick:
        out = RESULTS_DIR / "fraud_baselines.csv"
        df.to_csv(out, index=False)
        print(f"\nSaved {len(df)} rows to {out}")
    print("\nMean over seeds:")
    print(df.groupby(["ratio", "model", "alpha"])[["fnr", "fpr", "oracle_fpr", "auroc"]]
            .mean().round(4))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="1 seed, 5%% ratio, no CSV")
    main(p.parse_args().quick)