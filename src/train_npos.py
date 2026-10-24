"""Experiment 3 (protocol v3): does CRT's remaining harm shrink with more positives?

Lending Club only. Same sweep, splits, models and evaluation as v2/v3, but with a
different number of defaults kept per seed (LC_N_POS): 500 or 5,000 (1,500 is the
existing v2/v3 run). Trains, per (seed, ratio):
  A   plain BCE (early stopping), as in train_crt.py
  C   original CRT (probabilities, T = 0.1), lr 1e-3, per alpha    (train_crt.finetune)
  Cs  scale-free CRT (logits, T = 0.1 x std), lr 1e-3, per alpha   (train_crt_v3.finetune_scaled)
and scores all three with the same 100 paired re-splits.

Pre-specified (Oct 24, before this run):
  Question  Is the CRT - A harm smaller with 5,000 positives than with 500?
  Primary   per seed, average Cs - A over the 4 ratios (alpha = 0.10); compare
            N = 500 vs N = 5,000 with a Welch t-test (different data, so unpaired).
            Same for C - A. "Overfitting supported" if the harm is significantly
            smaller at 5,000 (p < 0.05); "not supported" otherwise.
  Fallback  if N = 5,000 is too slow on the CPU, run ratios 5% and 10% only
            (where Cs - A was significant at 1,500) and say so.

Usage (from the project root):
    python src/train_npos.py --n_pos 500 --quick              # seed 0, 5% only, no CSV
    python src/train_npos.py --n_pos 500                      # -> results/lending_club_npos500.csv
    python src/train_npos.py --n_pos 5000
    python src/train_npos.py --n_pos 5000 --ratios 0.05 0.10  # fallback
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

import data
from config import RESULTS_DIR, RANDOM_SEEDS, IMBALANCE_RATIOS, TARGET_FNR_LEVELS, VAL_FRAC, PRIMARY_FT_LR
from baselines import to_np, train_bce, evaluate
from train_crt import finetune
from train_crt_v3 import finetune_scaled, PRIMARY_SCALE


def main(n_pos, seeds, ratios, quick):
    data.DATASETS["lending_club"]["n_pos"] = n_pos           # defaults kept per seed
    lr = PRIMARY_FT_LR
    rows = []
    for seed in seeds:
        sweep = data.get_imbalance_sweep_splits(seed, ratios=ratios, dataset="lending_club")
        for ratio, (train, cal, test) in sweep.items():
            t0 = time.time()
            X_tr, y_tr = to_np(*train)
            cal, test = to_np(*cal), to_np(*test)
            X_fit, X_val, y_fit, y_val = train_test_split(       # identical to train_crt.py
                X_tr, y_tr, test_size=VAL_FRAC, stratify=y_tr, random_state=seed)
            fit = (torch.from_numpy(X_fit), torch.from_numpy(y_fit))
            val = (torch.from_numpy(X_val), torch.from_numpy(y_val))

            model_a = train_bce(X_fit, y_fit, class_weighted=False, seed=seed, val=(X_val, y_val))
            entries = [("A", model_a, -1, None)]
            for alpha in TARGET_FNR_LEVELS:
                m, _, ep = finetune(model_a, fit, val, lr, seed, "crt", alpha)
                entries.append(("C", m, ep, alpha))
                m, _, ep = finetune_scaled(model_a, fit, val, lr, seed, alpha, PRIMARY_SCALE)
                entries.append(("Cs", m, ep, alpha))

            for label, model, ep, only_alpha in entries:
                for r in evaluate(model, cal, test, seed):
                    if only_alpha is not None and r["alpha"] != only_alpha:
                        continue
                    rows.append(dict(dataset="lending_club", n_pos=n_pos, model=label, seed=seed,
                                     ratio=ratio, best_epoch=ep, **r))
                    print(f"N={n_pos} seed={seed} ratio={ratio:.2f} model={label:<2} "
                          f"alpha={r['alpha']:.2f} | FNR={r['fnr']:.3f} FPR={r['fpr']:.4f} "
                          f"AUROC={r['auroc']:.4f} ep={ep}")
            print(f"  ({time.time() - t0:.0f}s)")

    df = pd.DataFrame(rows)
    if not quick:
        out = RESULTS_DIR / f"lending_club_npos{n_pos}.csv"
        df.to_csv(out, index=False)
        print(f"\nSaved {len(df)} rows to {out}")
    print("\nMean over seeds:")
    print(df.groupby(["ratio", "alpha", "model"])[["fnr", "fpr", "auroc"]].mean().round(4).to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_pos", type=int, required=True, help="defaults kept per seed, e.g. 500 or 5000")
    p.add_argument("--ratios", type=float, nargs="+", default=IMBALANCE_RATIOS)
    p.add_argument("--seeds", type=int, nargs="+", default=RANDOM_SEEDS)
    p.add_argument("--quick", action="store_true", help="seed 0, 5%% ratio, no CSV")
    a = p.parse_args()
    if a.quick:
        main(a.n_pos, [0], [0.05], True)
    else:
        main(a.n_pos, a.seeds, a.ratios, False)