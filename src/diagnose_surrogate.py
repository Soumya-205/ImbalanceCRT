"""Experiment 0 (protocol v3): does CRT's smooth FPR track the real FPR?

Descriptive diagnostic: no CRT training. For Model A (trained exactly as in
train_crt.py, so it is the same model C starts from), compute one CRT step's
quantities on the FIT data:

  lambda      soft CRC threshold on a random half of the FIT positives (alpha = 0.10)
  hard_fpr    real share of FIT negatives at or above lambda
  smooth_fpr  CRT's differentiable stand-in: mean sigmoid((score - lambda) / T)
  focus       share of the loss gradient that falls on the 10% of negatives closest
              to lambda (0.10 = no focus at all, i.e. every negative pushed equally)

Two versions of the score space:
  prob   probabilities, T = 0.1  (Yeh et al.; what Model C used)
  logit  logits,        T = 1.0  (candidate fix for Experiment 1)

Hypothesis (pre-specified, Oct 24): in prob space with T = 0.1 the stand-in is a poor
copy of the real FPR at low positive ratios: when lambda is near 0 (weak separation,
e.g. Lending Club) every negative sits near the line, smooth_fpr drifts towards 0.5
and focus stays near 0.10, so the gradient pushes all negatives about equally instead
of the ones that cause false alarms. This uses FIT data only (no cal/test), so its
results may be used to design Experiment 1.

Usage (from the project root):
    python src/diagnose_surrogate.py                          # fraud, seeds 0-2
    python src/diagnose_surrogate.py --dataset lending_club
    python src/diagnose_surrogate.py --seeds 0                # quick check
"""
import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from config import RESULTS_DIR, IMBALANCE_RATIOS, VAL_FRAC, CRT_T
from data import get_imbalance_sweep_splits, DATASETS
from baselines import to_np, train_bce
from train_crt import soft_crc_lambda, smooth_fpr

ALPHA = 0.10        # primary target FNR
T_LOGIT = 1.0       # temperature for the logit-space version (pre-specified)
NEAR_FRAC = 0.10    # "near the line" = the 10% of negatives closest to lambda
SCALES = [0.05, 0.1, 0.2]   # Exp 1 candidates: logit space, T = scale x std of negative logits


@torch.no_grad()
def surrogate_stats(model, X, y, seed, alpha=ALPHA):
    """One CRT step's lambda, real FPR, smooth FPR and gradient focus, in both spaces."""
    model.eval()
    z = model(X)                         # logits
    p = torch.sigmoid(z)                 # probabilities (what CRT used)
    pos = torch.nonzero(y == 1).squeeze(1)
    neg = torch.nonzero(y == 0).squeeze(1)
    g = torch.Generator().manual_seed(seed)
    cal = pos[torch.randperm(len(pos), generator=g)][: len(pos) // 2]   # as in a CRT step

    out = {"neg_prob_median": p[neg].median().item(),
           "pos_prob_median": p[pos].median().item()}
    spaces = [("prob", p, CRT_T), ("logit", z, T_LOGIT)]
    sd = z[neg].std().item()                      # spread of negative logits
    spaces += [(f"lsc{c:g}", z, c * sd) for c in SCALES]
    for space, s, T in spaces:
        lam = soft_crc_lambda(s[cal], alpha)
        sn = s[neg]
        u = torch.sigmoid((sn - lam) / T)
        w = u * (1 - u) / T                       # gradient of each negative's term w.r.t. its score
        dist = (sn - lam).abs()
        near = dist <= torch.quantile(dist, NEAR_FRAC)
        out.update({
            f"{space}_lambda": lam.item(),
            f"{space}_hard_fpr": (sn >= lam).float().mean().item(),
            f"{space}_smooth_fpr": smooth_fpr(sn, lam, T).item(),
            f"{space}_focus": (w[near].sum() / w.sum()).item(),
            f"{space}_T": T,
        })
    return out


def main(dataset, seeds):
    rows = []
    for seed in seeds:
        sweep = get_imbalance_sweep_splits(seed, ratios=IMBALANCE_RATIOS, dataset=dataset)
        for ratio, (train, _, _) in sweep.items():
            X_tr, y_tr = to_np(*train)
            # identical FIT/VAL split and Model A as in train_crt.py
            X_fit, X_val, y_fit, y_val = train_test_split(
                X_tr, y_tr, test_size=VAL_FRAC, stratify=y_tr, random_state=seed)
            model_a = train_bce(X_fit, y_fit, class_weighted=False, seed=seed, val=(X_val, y_val))
            st = surrogate_stats(model_a, torch.from_numpy(X_fit), torch.from_numpy(y_fit), seed)
            rows.append(dict(dataset=dataset, seed=seed, ratio=ratio, **st))
            print(f"seed={seed} ratio={ratio:.2f} | real FPR={st['prob_hard_fpr']:.3f} | "
                  f"prob: lambda={st['prob_lambda']:.4f} smooth={st['prob_smooth_fpr']:.3f} "
                  f"focus={st['prob_focus']:.2f} | "
                  f"logit: smooth={st['logit_smooth_fpr']:.3f} focus={st['logit_focus']:.2f}")

    df = pd.DataFrame(rows)
    out = RESULTS_DIR / f"{dataset}_surrogate_diag.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df)} rows to {out}")
    cols = ["neg_prob_median", "prob_lambda", "prob_hard_fpr", "prob_smooth_fpr", "prob_focus",
            "logit_hard_fpr", "logit_smooth_fpr", "logit_focus"]
    print("\nMean over seeds:")
    print(df.groupby("ratio")[cols].mean().round(4).to_string())
    sc = [f"{n}_{m}" for c in SCALES for n in [f"lsc{c:g}"] for m in ("T", "smooth_fpr", "focus")]
    print("\nExp 1 candidates (logit space, T = scale x std of negative logits), mean over seeds:")
    print(df.groupby("ratio")[["prob_hard_fpr"] + sc].mean().round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS), default="fraud")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    main(a.dataset, a.seeds)