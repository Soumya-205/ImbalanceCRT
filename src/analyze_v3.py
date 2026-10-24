"""Experiment 1 statistics: scale-free CRT (Cs) vs Model A and vs original CRT (C).

Reads results/{dataset}_crt_v3.csv (A and Cs) and results/{dataset}_crt_v2.csv
(A and C at lr = 1e-3). Writes results/stats_v3.csv.

Per (dataset, alpha, scale, ratio), over the 10 paired seeds:
  Cs - A   : FPR change vs post-hoc CRC (both from the v3 run, same re-splits)
  Cs - C   : FPR change vs original CRT, computed as (Cs - A_v3) - (C - A_v2), so it
             stays valid even if the retrained A differs slightly from the v2 A
Mean in percentage points, t-based 95% CI, Wilcoxon p, Holm over the 4 ratios.
Also checks how closely the retrained A matches the v2 A.

Usage (from the project root):  python src/analyze_v3.py
"""
import numpy as np
import pandas as pd

from config import RESULTS_DIR, PRIMARY_FT_LR
from analyze import holm
from scipy.stats import t as t_dist, wilcoxon

KEY = ["alpha", "ratio", "seed"]
PRIMARY_SCALE = 0.1


def summarise(d):
    """d: paired differences (fraction units) -> mean pp, 95% CI, Wilcoxon p."""
    d = d.dropna() * 100
    n = len(d)
    half = t_dist.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
    p = wilcoxon(d, zero_method="zsplit").pvalue if (d != 0).any() else 1.0
    return dict(n=n, d_fpr_pp=d.mean(), ci_lo=d.mean() - half, ci_hi=d.mean() + half,
                wins=int((d < 0).sum()), p=p)


def run(dataset):
    f3, f2 = RESULTS_DIR / f"{dataset}_crt_v3.csv", RESULTS_DIR / f"{dataset}_crt_v2.csv"
    if not f3.exists():
        print(f"(missing {f3.name}; skipped)")
        return []
    v3, v2 = pd.read_csv(f3), pd.read_csv(f2)
    a3 = v3[v3.model == "A"].set_index(KEY)
    a2 = v2[v2.model == "A"].set_index(KEY)
    c2 = v2[(v2.model == "C") & np.isclose(v2.lr, PRIMARY_FT_LR)].set_index(KEY)

    diff = (a3.fpr - a2.fpr.reindex(a3.index)).abs()
    print(f"\n{dataset}: retrained A vs v2 A, max |FPR difference| = {diff.max() * 100:.4f} pp "
          f"({'identical' if diff.max() < 1e-12 else 'not identical: Cs - C uses the A-relative form'})")

    out = []
    for scale, g in v3[v3.model == "Cs"].groupby("scale"):
        g = g.set_index(KEY)
        d_a = g.fpr - a3.fpr.reindex(g.index)                          # Cs - A
        d_c = d_a - (c2.fpr - a2.fpr).reindex(g.index)                 # Cs - C
        dauc = g.auroc - a3.auroc.reindex(g.index)
        for (alpha, ratio), idx in g.groupby(level=["alpha", "ratio"]).groups.items():
            for comp, d in (("Cs-A", d_a), ("Cs-C", d_c)):
                out.append(dict(dataset=dataset, scale=scale, alpha=alpha, ratio=ratio,
                                comparison=comp, d_auroc=dauc.loc[idx].mean(),
                                mean_best_epoch=g.loc[idx, "best_epoch"].mean(),
                                **summarise(d.loc[idx])))
    return out


def main():
    st = pd.DataFrame(sum((run(ds) for ds in ["fraud", "lending_club"]), []))
    if st.empty:
        return
    st["p_holm"] = np.nan
    for _, idx in st.groupby(["dataset", "scale", "alpha", "comparison"]).groups.items():
        st.loc[idx, "p_holm"] = holm(st.loc[idx, "p"])
    st = st.sort_values(["dataset", "comparison", "scale", "alpha", "ratio"]).reset_index(drop=True)
    st.to_csv(RESULTS_DIR / "stats_v3.csv", index=False)
    print(f"\nwrote stats_v3.csv ({len(st)} comparisons)")
    cols = ["dataset", "comparison", "alpha", "ratio", "d_fpr_pp", "ci_lo", "ci_hi",
            "wins", "p_holm", "d_auroc", "mean_best_epoch"]
    print(f"\nPRIMARY (scale = {PRIMARY_SCALE}), FPR change in percentage points:")
    print(st[np.isclose(st.scale, PRIMARY_SCALE)][cols].round(3).to_string(index=False))
    print("\nSENSITIVITY (other scales), alpha = 0.10:")
    sens = st[~np.isclose(st.scale, PRIMARY_SCALE) & np.isclose(st.alpha, 0.10)]
    print(sens[["scale"] + cols].round(3).to_string(index=False))

    # pre-specified success check: Lending Club, alpha = 0.10, ratios 1/5/10%
    lc = st[(st.dataset == "lending_club") & np.isclose(st.scale, PRIMARY_SCALE)
            & np.isclose(st.alpha, 0.10) & st.ratio.isin([0.01, 0.05, 0.10])]
    if not lc.empty:
        not_worse = lc[lc.comparison == "Cs-A"].apply(
            lambda r: not (r.p_holm < 0.05 and r.d_fpr_pp > 0), axis=1).all()
        better_c = lc[lc.comparison == "Cs-C"].apply(
            lambda r: r.p_holm < 0.05 and r.d_fpr_pp < 0, axis=1).all()
        beats_a = lc[lc.comparison == "Cs-A"].apply(
            lambda r: r.p_holm < 0.05 and r.d_fpr_pp < 0, axis=1).all()
        print(f"\nSuccess criterion (LC, alpha 0.10, 1/5/10%): not worse than A = {not_worse}, "
              f"better than C = {better_c} -> {'MET' if not_worse and better_c else 'NOT MET'}"
              f"; beats post-hoc CRC at all three = {beats_a}")


if __name__ == "__main__":
    main()