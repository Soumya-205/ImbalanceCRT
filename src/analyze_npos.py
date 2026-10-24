"""Experiment 3 statistics: CRT harm vs number of positives (Lending Club).

Reads results/lending_club_npos500.csv and results/lending_club_npos5000.csv, and
takes N = 1,500 from the existing runs (A and C at lr 1e-3 from lending_club_crt_v2.csv,
Cs at scale 0.1 from lending_club_crt_v3.csv). Writes results/stats_npos.csv.

  Per (N, alpha, ratio): C - A and Cs - A, mean pp, t-based 95% CI, Wilcoxon p.
  Primary test (pre-specified): per seed, average the model - A difference over the
  ratios; compare N = 500 vs N = 5,000 with a Welch t-test (alpha = 0.10).

Usage (from the project root):  python src/analyze_npos.py
"""
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

from config import RESULTS_DIR, PRIMARY_FT_LR
from analyze_v3 import summarise

KEY = ["alpha", "ratio", "seed"]


def load():
    frames = []
    for n in (500, 5000):
        f = RESULTS_DIR / f"lending_club_npos{n}.csv"
        if f.exists():
            frames.append(pd.read_csv(f)[["n_pos", "model", *KEY, "fpr", "auroc"]])
        else:
            print(f"(missing {f.name}; skipped)")
    v2 = pd.read_csv(RESULTS_DIR / "lending_club_crt_v2.csv")
    v2 = v2[(v2.model == "A") | ((v2.model == "C") & np.isclose(v2.lr, PRIMARY_FT_LR))]
    frames.append(v2.assign(n_pos=1500)[["n_pos", "model", *KEY, "fpr", "auroc"]])
    f3 = RESULTS_DIR / "lending_club_crt_v3.csv"
    if f3.exists():
        v3 = pd.read_csv(f3)
        v3 = v3[(v3.model == "Cs") & np.isclose(v3.scale, 0.1)]
        frames.append(v3.assign(n_pos=1500)[["n_pos", "model", *KEY, "fpr", "auroc"]])
    return pd.concat(frames, ignore_index=True)


def main():
    df = load()
    a = df[df.model == "A"].set_index(["n_pos", *KEY])
    diffs, rows = [], []
    for (n, model), g in df[df.model != "A"].groupby(["n_pos", "model"]):
        g = g.set_index(["n_pos", *KEY])
        d = (g.fpr - a.fpr.reindex(g.index)).rename("d").reset_index()
        d["dauc"] = (g.auroc - a.auroc.reindex(g.index)).values
        d["model"] = model
        diffs.append(d)
        for (alpha, ratio), dd in d.groupby(["alpha", "ratio"]):
            rows.append(dict(n_pos=n, model=model, alpha=alpha, ratio=ratio,
                             d_auroc=dd.dauc.mean(), **summarise(dd.d)))
    st = pd.DataFrame(rows).sort_values(["model", "alpha", "ratio", "n_pos"])
    st.to_csv(RESULTS_DIR / "stats_npos.csv", index=False)
    print(f"wrote stats_npos.csv ({len(st)} rows)")
    print("\nFPR change vs A (pp) by number of positives, alpha = 0.10:")
    view = st[np.isclose(st.alpha, 0.10)]
    print(view.pivot_table(index=["model", "ratio"], columns="n_pos", values="d_fpr_pp").round(2).to_string())
    print("\nSame, with 95% CI and p:")
    print(view[["model", "ratio", "n_pos", "d_fpr_pp", "ci_lo", "ci_hi", "wins", "p", "d_auroc"]]
          .round(3).to_string(index=False))

    # primary test: per-seed mean over ratios, N = 500 vs N = 5,000 (Welch)
    d = pd.concat(diffs)
    d = d[np.isclose(d.alpha, 0.10)]
    print("\nPRIMARY: per-seed mean of (model - A) over ratios, alpha = 0.10")
    for model, g in d.groupby("model"):
        per_seed = g.groupby(["n_pos", "seed"]).d.mean() * 100
        means = per_seed.groupby(level="n_pos").agg(["mean", "std", "count"]).round(3)
        print(f"\n{model}:\n{means.to_string()}")
        if {500, 5000} <= set(per_seed.index.get_level_values("n_pos")):
            x, y = per_seed.loc[500], per_seed.loc[5000]
            t = ttest_ind(x, y, equal_var=False)
            if t.pvalue < 0.05 and y.mean() < x.mean():
                verdict = "significantly smaller at 5,000 (overfitting supported)"
            elif t.pvalue < 0.05:
                verdict = "significantly LARGER at 5,000 (overfitting not supported)"
            else:
                verdict = "no significant change (overfitting not supported)"
            print(f"  N=500 vs N=5000: diff {x.mean() - y.mean():+.2f} pp, Welch p = {t.pvalue:.3f} -> harm {verdict}")


if __name__ == "__main__":
    main()