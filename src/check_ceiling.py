"""Is the high Lending Club FPR a property of the data or of our MLP?

Trains a gradient-boosted tree model (scikit-learn HistGradientBoostingClassifier,
default settings) on exactly the same training splits as Model A, scores it with the
same post-hoc CRC re-split evaluation (baselines.evaluate), and compares its AUROC and
FPR with Model A from results/{dataset}_crt_v2.csv. Also prints the FPR that a
binormal ROC curve with the same AUROC would give at the target FNR:
    FPR ~ Phi( Phi^-1(1 - alpha) - sqrt(2) * Phi^-1(AUROC) )   (approximation)

Descriptive check (seeds 0-2), not part of the pre-specified protocol.

Usage (from the project root):
    python src/check_ceiling.py                     # Lending Club, seeds 0-2
    python src/check_ceiling.py --dataset fraud
"""
import argparse

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier

from config import RESULTS_DIR, IMBALANCE_RATIOS
from data import get_imbalance_sweep_splits, DATASETS
from baselines import to_np, evaluate


class TreeScorer:
    """Lets baselines.evaluate score a sklearn model: returns its raw decision scores."""
    def __init__(self, clf):
        self.clf = clf

    def eval(self):
        return self

    def __call__(self, X):
        return torch.from_numpy(self.clf.decision_function(X.numpy()).astype(np.float32))


def binormal_fpr(auroc, alpha):
    return norm.cdf(norm.ppf(1 - alpha) - np.sqrt(2) * norm.ppf(auroc))


def main(dataset, seeds):
    rows = []
    for seed in seeds:
        sweep = get_imbalance_sweep_splits(seed, ratios=IMBALANCE_RATIOS, dataset=dataset)
        for ratio, (train, cal, test) in sweep.items():
            X_tr, y_tr = to_np(*train)
            clf = HistGradientBoostingClassifier(random_state=seed).fit(X_tr, y_tr)
            for r in evaluate(TreeScorer(clf), to_np(*cal), to_np(*test), seed):
                rows.append(dict(dataset=dataset, model="GBT", seed=seed, ratio=ratio, **r))
            print(f"seed={seed} ratio={ratio:.2f} | GBT AUROC={rows[-1]['auroc']:.4f} "
                  f"FPR@10%={rows[-1]['fpr']:.4f}")

    gbt = pd.DataFrame(rows)
    v2 = pd.read_csv(RESULTS_DIR / f"{dataset}_crt_v2.csv")
    a = v2[(v2.model == "A") & v2.seed.isin(seeds)]
    both = pd.concat([gbt, a[gbt.columns.intersection(a.columns)]], ignore_index=True)
    out = both.groupby(["alpha", "ratio", "model"])[["fnr", "fpr", "oracle_fpr", "auroc"]].mean()
    out["binormal_fpr"] = binormal_fpr(out.auroc, out.index.get_level_values("alpha"))
    gbt.to_csv(RESULTS_DIR / f"{dataset}_ceiling_check.csv", index=False)
    print(f"\nSaved {len(gbt)} rows to {dataset}_ceiling_check.csv")
    print(f"\n{dataset}: gradient-boosted trees (GBT) vs MLP Model A, mean over seeds {seeds}:")
    print((out * 100).round(2).rename(columns={"auroc": "auroc_x100"}).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS), default="lending_club")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    main(a.dataset, a.seeds)