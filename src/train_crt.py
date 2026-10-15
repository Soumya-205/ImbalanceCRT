"""Model C: conformal risk training (CRT; Yeh et al., NeurIPS 2025, Algorithm 2,
FNR case), plus control A+ (same fine-tuning protocol with plain BCE).

Follows the official implementation (github.com/chrisyeh96/conformal-risk-training,
run_polyp.py: e2ecrc, train_base, analysis/polyps.ipynb). Differences are marked
"DEVIATION" and must be reported in the paper.

Per (seed, ratio):
  1. Split the training data into FIT (1 - VAL_FRAC) and VAL (VAL_FRAC), stratified.
  2. Train Models A and B on FIT with early stopping on VAL loss (converged at every
     ratio). VAL must be unseen by every model, otherwise early stopping is useless.
  3. A+ : fine-tune A with BCE, once per LR in FT_LRS.
  4. C  : for each alpha, fine-tune A with CRT, once per LR in FT_LRS.
  5. Score every model with baselines.evaluate (same 100 re-splits -> paired).
     Every LR is saved. Column `selected` marks the LR with the best val metric
     (Yeh's protocol: val FPR for CRT, val loss for the BCE model).

Protocols (fixed Oct 15, before the v2 run):
  PRIMARY     : A, B, and A+/C at PRIMARY_FT_LR (no LR selection).
  SENSITIVITY : A+/C at every other LR, and Yeh-style val-selected LR.
  Why: v1 showed that with only ~49 val positives, val-based LR selection mostly
  picked lr=1e-2, which then did worst on held-out data (selection noise).

Usage (from the project root):
    python src/train_crt.py --quick    # seed 0, 5% ratio, prints only
    python src/train_crt.py            # full sweep -> results/fraud_crt_v2.csv
"""
import argparse
import copy
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from config import (RESULTS_DIR, RANDOM_SEEDS, IMBALANCE_RATIOS, TARGET_FNR_LEVELS,
                    BATCH_SIZE, CRT_T, FT_LRS, FT_MAX_EPOCHS, FT_PATIENCE,
                    FT_STEPS_PER_EPOCH, CRT_NEG_PER_STEP, VAL_FRAC, PRIMARY_FT_LR)
from crc import crc_threshold_fnr, fnr_fpr
from data import get_imbalance_sweep_splits
from baselines import to_np, train_bce, evaluate


# ---------------------------------------------------------------- CRT pieces
def soft_crc_lambda(pos_scores, alpha):
    """Differentiable in-batch CRC threshold (Algorithm 2) with Yeh et al.'s
    variance reduction: average the scores in a small window around the
    order statistic ("use up to 0.5% of true-positive pixels").

    pos_scores: probabilities of the pseudo-calibration positives (requires grad).
    Returns a 0-dim tensor, or None if CRC is infeasible for this many positives.
    """
    n = pos_scores.numel()
    k = int(np.floor(alpha * (n + 1) - 1))       # same order statistic as crc.py
    if k < 0:
        return None
    s_sorted, _ = torch.sort(pos_scores)         # ascending; gradients flow to the chosen scores
    h = int(np.ceil(n / 400))                    # Yeh: half-window = ceil(n/400)
    lo, hi = max(0, k - h + 1), min(n, k + h + 1)  # mirror of Yeh's window (they sort descending)
    return s_sorted[lo:hi].mean()


def smooth_fpr(neg_scores, lam, T=CRT_T):
    """Yeh et al. fpr_loss: mean of sigmoid((score - lambda) / T) over negatives."""
    return torch.sigmoid((neg_scores - lam) / T).mean()


@torch.no_grad()
def val_fpr_at_crc(model, X, y, alpha):
    """Yeh's per-epoch validation metric: CRC lambda on the val set, FPR on the same set."""
    model.eval()
    s = model(X).numpy()
    yn = y.numpy()
    lam, ok = crc_threshold_fnr(s[yn == 1], alpha)
    return fnr_fpr(yn, s, lam)[1] if ok else np.inf


@torch.no_grad()
def val_bce(model, X, y):
    model.eval()
    return nn.functional.binary_cross_entropy_with_logits(model(X), y).item()


# ------------------------------------------------------------ fine-tuning loop
def finetune(model_a, fit, val, lr, seed, mode, alpha=None):
    """Fine-tune a copy of Model A. mode='crt' (Model C) or 'bce' (control A+).
    Early stopping exactly as Yeh: keep the best-val epoch, stop after more than
    FT_PATIENCE epochs without improvement, at most FT_MAX_EPOCHS.
    Returns (best_model, best_val_metric, best_epoch)."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    model = copy.deepcopy(model_a)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xf, yf = fit
    Xv, yv = val
    pos_idx = torch.nonzero(yf == 1).squeeze(1)
    neg_idx = torch.nonzero(yf == 0).squeeze(1)

    best_val, best_epoch, best_state, since = np.inf, -1, None, 0
    for epoch in range(FT_MAX_EPOCHS):
        model.train()
        for _ in range(FT_STEPS_PER_EPOCH):
            if mode == "crt":
                # DEVIATION (decided Oct 14): batches with a FIXED number of positives.
                # A random half of the fit positives is the pseudo-calibration set
                # (Yeh: first half of a random batch). The FPR loss only uses negatives,
                # so the negatives are drawn separately; lambda and the loss still come
                # from disjoint data, as in Algorithm 2.
                perm = pos_idx[torch.randperm(len(pos_idx), generator=g)]
                cal_pos = perm[: len(perm) // 2]
                negs = neg_idx[torch.randint(len(neg_idx), (CRT_NEG_PER_STEP,), generator=g)]
                lam = soft_crc_lambda(torch.sigmoid(model(Xf[cal_pos])), alpha)
                if lam is None:
                    raise ValueError(f"CRC infeasible with {len(cal_pos)} pseudo-cal positives")
                loss = smooth_fpr(torch.sigmoid(model(Xf[negs])), lam)   # Yeh: probabilities, T=0.1
            else:
                # A+: ordinary BCE on natural random batches (Yeh's cross-entropy baseline)
                idx = torch.randint(len(yf), (BATCH_SIZE,), generator=g)
                loss = nn.functional.binary_cross_entropy_with_logits(model(Xf[idx]), yf[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

        v = val_fpr_at_crc(model, Xv, yv, alpha) if mode == "crt" else val_bce(model, Xv, yv)
        since += 1
        if v < best_val:
            best_val, best_epoch, since = v, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        if since > FT_PATIENCE:
            break

    model.load_state_dict(best_state)
    return model, best_val, best_epoch


def run_grid(model_a, fit, val, seed, mode, alpha=None):
    """Fine-tune once per LR in FT_LRS. Returns a list of (model, info) for every LR;
    info['selected'] is True for the LR with the lowest val metric (Yeh's protocol)."""
    runs = []
    for lr in FT_LRS:
        model, v, ep = finetune(model_a, fit, val, lr, seed, mode, alpha)
        runs.append((model, dict(lr=lr, val_metric=v, best_epoch=ep)))
    best = min(range(len(runs)), key=lambda i: runs[i][1]["val_metric"])
    for i, (_, info) in enumerate(runs):
        info["selected"] = (i == best)
    return runs


# ----------------------------------------------------------------------- main
def main(quick):
    seeds = [0] if quick else RANDOM_SEEDS
    ratios = [0.05] if quick else IMBALANCE_RATIOS
    rows = []
    for seed in seeds:
        sweep = get_imbalance_sweep_splits(seed, ratios=ratios)
        for ratio, (train, cal, test) in sweep.items():
            t0 = time.time()
            X_tr, y_tr = to_np(*train)
            cal, test = to_np(*cal), to_np(*test)

            # 1. DEVIATION: Yeh use the calibration ("val") set for early stopping and
            #    LR choice; we hold out part of TRAIN instead, so cal/test stay untouched.
            X_fit, X_val, y_fit, y_val = train_test_split(
                X_tr, y_tr, test_size=VAL_FRAC, stratify=y_tr, random_state=seed)
            fit = (torch.from_numpy(X_fit), torch.from_numpy(y_fit))
            val = (torch.from_numpy(X_val), torch.from_numpy(y_val))

            # 2. Models A and B on FIT with early stopping on VAL loss.
            model_a = train_bce(X_fit, y_fit, class_weighted=False, seed=seed, val=(X_val, y_val))
            model_b = train_bce(X_fit, y_fit, class_weighted=True, seed=seed, val=(X_val, y_val))

            base = dict(lr=np.nan, val_metric=np.nan, best_epoch=-1, selected=True)
            entries = [("A", model_a, base, None), ("B", model_b, base, None)]
            # 3. A+ (does not depend on alpha)
            entries += [("A+", m, info, None) for m, info in run_grid(model_a, fit, val, seed, "bce")]
            # 4. C, one model per alpha and LR (CRT trains for a specific target)
            for alpha in TARGET_FNR_LEVELS:
                entries += [("C", m, info, alpha)
                            for m, info in run_grid(model_a, fit, val, seed, "crt", alpha)]

            # 5. Same evaluation for all (paired re-splits)
            for label, model, info, only_alpha in entries:
                for r in evaluate(model, cal, test, seed):
                    if only_alpha is not None and r["alpha"] != only_alpha:
                        continue                         # a C model is only scored at its own alpha
                    rows.append(dict(dataset="fraud", model=label, seed=seed, ratio=ratio,
                                     **r, **info))
                    if label in ("A", "B") or info["lr"] == PRIMARY_FT_LR:
                        print(f"seed={seed} ratio={ratio:.2f} model={label:<2} alpha={r['alpha']:.2f} | "
                              f"FNR={r['fnr']:.3f} FPR={r['fpr']:.4f} oracleFPR={r['oracle_fpr']:.4f} "
                              f"AUROC={r['auroc']:.4f} | lr={info['lr']:.0e} ep={info['best_epoch']}")
            print(f"  ({time.time() - t0:.0f}s)")

    df = pd.DataFrame(rows)
    if not quick:
        out = RESULTS_DIR / "fraud_crt_v2.csv"
        df.to_csv(out, index=False)
        print(f"\nSaved {len(df)} rows to {out}")
    cols = ["fnr", "fpr", "oracle_fpr", "auroc"]
    primary = df[df.model.isin(["A", "B"]) | (df.lr == PRIMARY_FT_LR)]
    print(f"\nPRIMARY protocol (A+/C at lr={PRIMARY_FT_LR:g}), mean over seeds:")
    print(primary.groupby(["ratio", "model", "alpha"])[cols].mean().round(4).to_string())
    sel = df[df.selected]
    print("\nSENSITIVITY: Yeh-style val-selected LR, mean over seeds:")
    print(sel.groupby(["ratio", "model", "alpha"])[cols].mean().round(4).to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="seed 0, 5%% ratio, no CSV")
    main(p.parse_args().quick)