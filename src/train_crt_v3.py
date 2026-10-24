"""Experiment 1 (protocol v3): CRT with a scale-free temperature (model "Cs").

What changes compared with Model C (train_crt.py), and nothing else:
  * the CRC threshold and the smooth FPR are computed on LOGITS, not probabilities;
  * the temperature is T = scale x std(logits of the step's negatives), recomputed at
    every step and detached (no gradient flows through T).
Everything else is identical to Model C: start from Model A, Adam, the same fixed
positives per step (DEVIATION 1), 256 negatives, 20 steps per epoch, early stopping
on Yeh's val metric (CRC threshold and FPR on VAL), patience 10, max 100 epochs.

Why: Experiment 0 (diagnose_surrogate.py) showed that on Lending Club, Yeh's loss
(probabilities, T = 0.1) pushes every negative about equally at low positive ratios
(gradient focus 0.10 = none), and that a fixed T in logit space is no better. A
temperature set relative to the spread of the scores keeps the loss focused near
the threshold at every ratio (focus about 0.63 at scale 0.1, on FIT data only).

Pre-specified (Oct 24, before this run):
  PRIMARY      scale = 0.1, lr = 1e-3 (same lr as v2 primary), alpha in {0.05, 0.10}
  SENSITIVITY  scale = 0.05 and 0.2 at lr = 1e-3
  Success      Lending Club, alpha = 0.10, ratios 1/5/10%: Cs - A not significantly
               above 0 (Holm) AND Cs significantly below C (Holm).

Model A is retrained exactly as in train_crt.py (same seed, same FIT data), so it
should match the A in {dataset}_crt_v2.csv; analyze_v3.py checks this.

Usage (from the project root):
    python src/train_crt_v3.py --quick                       # fraud, seed 0, 5% ratio, no CSV
    python src/train_crt_v3.py --dataset lending_club --quick
    python src/train_crt_v3.py                               # -> results/fraud_crt_v3.csv
    python src/train_crt_v3.py --dataset lending_club        # -> results/lending_club_crt_v3.csv
"""
import argparse
import copy
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from config import (RESULTS_DIR, RANDOM_SEEDS, IMBALANCE_RATIOS, TARGET_FNR_LEVELS,
                    FT_MAX_EPOCHS, FT_PATIENCE, FT_STEPS_PER_EPOCH, CRT_NEG_PER_STEP,
                    VAL_FRAC, PRIMARY_FT_LR)
from data import get_imbalance_sweep_splits, DATASETS
from baselines import to_np, train_bce, evaluate
from train_crt import soft_crc_lambda, val_fpr_at_crc

SCALES = [0.1, 0.05, 0.2]       # first = primary
PRIMARY_SCALE = 0.1


def finetune_scaled(model_a, fit, val, lr, seed, alpha, scale):
    """Fine-tune a copy of Model A with logit-space CRT and a scale-free temperature.
    Same loop and early stopping as train_crt.finetune(mode='crt').
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
            perm = pos_idx[torch.randperm(len(pos_idx), generator=g)]
            cal_pos = perm[: len(perm) // 2]                      # same as Model C
            negs = neg_idx[torch.randint(len(neg_idx), (CRT_NEG_PER_STEP,), generator=g)]
            lam = soft_crc_lambda(model(Xf[cal_pos]), alpha)      # threshold on LOGITS
            if lam is None:
                raise ValueError(f"CRC infeasible with {len(cal_pos)} pseudo-cal positives")
            zn = model(Xf[negs])                                  # negative logits
            T = scale * zn.detach().std().clamp_min(1e-6)         # scale-free, no gradient
            loss = torch.sigmoid((zn - lam) / T).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

        v = val_fpr_at_crc(model, Xv, yv, alpha)                  # Yeh's val metric
        since += 1
        if v < best_val:
            best_val, best_epoch, since = v, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        if since > FT_PATIENCE:
            break

    model.load_state_dict(best_state)
    return model, best_val, best_epoch


def main(quick, dataset="fraud"):
    seeds = [0] if quick else RANDOM_SEEDS
    ratios = [0.05] if quick else IMBALANCE_RATIOS
    lr = PRIMARY_FT_LR
    rows = []
    for seed in seeds:
        sweep = get_imbalance_sweep_splits(seed, ratios=ratios, dataset=dataset)
        for ratio, (train, cal, test) in sweep.items():
            t0 = time.time()
            X_tr, y_tr = to_np(*train)
            cal, test = to_np(*cal), to_np(*test)
            X_fit, X_val, y_fit, y_val = train_test_split(       # identical to train_crt.py
                X_tr, y_tr, test_size=VAL_FRAC, stratify=y_tr, random_state=seed)
            fit = (torch.from_numpy(X_fit), torch.from_numpy(y_fit))
            val = (torch.from_numpy(X_val), torch.from_numpy(y_val))

            model_a = train_bce(X_fit, y_fit, class_weighted=False, seed=seed, val=(X_val, y_val))
            entries = [("A", model_a, dict(lr=np.nan, scale=np.nan, val_metric=np.nan,
                                           best_epoch=-1), None)]
            for alpha in TARGET_FNR_LEVELS:
                for scale in SCALES:
                    m, v, ep = finetune_scaled(model_a, fit, val, lr, seed, alpha, scale)
                    entries.append(("Cs", m, dict(lr=lr, scale=scale, val_metric=v,
                                                  best_epoch=ep), alpha))

            for label, model, info, only_alpha in entries:
                for r in evaluate(model, cal, test, seed):           # same paired re-splits
                    if only_alpha is not None and r["alpha"] != only_alpha:
                        continue
                    rows.append(dict(dataset=dataset, model=label, seed=seed, ratio=ratio,
                                     **r, **info))
                    if label == "A" or info["scale"] == PRIMARY_SCALE:
                        print(f"seed={seed} ratio={ratio:.2f} model={label:<2} alpha={r['alpha']:.2f} | "
                              f"FNR={r['fnr']:.3f} FPR={r['fpr']:.4f} AUROC={r['auroc']:.4f} | "
                              f"scale={info['scale']} ep={info['best_epoch']}")
            print(f"  ({time.time() - t0:.0f}s)")

    df = pd.DataFrame(rows)
    if not quick:
        out = RESULTS_DIR / f"{dataset}_crt_v3.csv"
        df.to_csv(out, index=False)
        print(f"\nSaved {len(df)} rows to {out}")
    print("\nMean over seeds (FPR, FNR, AUROC):")
    print(df.groupby(["ratio", "alpha", "model", "scale"], dropna=False)[["fnr", "fpr", "auroc"]]
            .mean().round(4).to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="seed 0, 5%% ratio, no CSV")
    p.add_argument("--dataset", choices=list(DATASETS), default="fraud")
    args = p.parse_args()
    main(args.quick, args.dataset)