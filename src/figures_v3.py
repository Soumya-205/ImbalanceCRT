"""Figure for protocol v3 (Lending Club): diagnosis and fix.

Top:    gradient focus of the CRT loss at the start of fine-tuning (Experiment 0):
        Yeh et al.'s loss (probabilities, T = 0.1) vs the scale-free loss
        (logits, T = 0.1 x std of negative logits). 0.10 = no focus.
Bottom: paired FPR change vs model A at target FNR 10% (mean, 95% CI, 10 seeds):
        original CRT (C, v2) and scale-free CRT (Cs, Experiment 1).

Reads results/lending_club_surrogate_diag.csv, results/stats_paired.csv,
results/stats_v3.csv. Writes results/figures/fig4_diagnosis_fix.(pdf|png).

Usage (from the project root):  python src/figures_v3.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import RESULTS_DIR, PRIMARY_FT_LR
from analyze import RATIOS, INK2, save          # also applies the paper's plot style

C_COL, CS_COL = "#2a78d6", "#eda100"            # categorical slots 1 and 4 (validated)


def main():
    diag = pd.read_csv(RESULTS_DIR / "lending_club_surrogate_diag.csv")
    st2 = pd.read_csv(RESULTS_DIR / "stats_paired.csv")
    st3 = pd.read_csv(RESULTS_DIR / "stats_v3.csv")
    c = st2[(st2.dataset == "lending_club") & (st2.model == "C") & np.isclose(st2.lr, PRIMARY_FT_LR)
            & np.isclose(st2.alpha, 0.10)].set_index("ratio").reindex(RATIOS)
    cs = st3[(st3.dataset == "lending_club") & (st3.comparison == "Cs-A") & np.isclose(st3.scale, 0.1)
             & np.isclose(st3.alpha, 0.10)].set_index("ratio").reindex(RATIOS)
    foc = diag.groupby("ratio")[["prob_focus", "lsc0.1_focus"]].mean().reindex(RATIOS)

    x = np.arange(len(RATIOS))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 3.9), sharex=True)

    ax1.axhline(0.10, color=INK2, linewidth=0.9, linestyle=(0, (3, 2)))
    ax1.text(1.5, 0.085, "0.10 = no focus", color=INK2, fontsize=6.5, ha="center", va="top")
    ax1.plot(x, foc["prob_focus"], marker="D", color=C_COL, linewidth=2, markersize=6,
             markeredgecolor="white", label="CRT loss (prob., $T$=0.1)")
    ax1.plot(x, foc["lsc0.1_focus"], marker="o", color=CS_COL, linewidth=2, markersize=6,
             markeredgecolor="white", label="Scale-free loss")
    ax1.set_ylim(0, 0.8)
    ax1.set_ylabel("Gradient focus")
    ax1.set_title("Lending Club: CRT loss at start of fine-tuning")
    ax1.legend(loc="center right", frameon=False)
    ax1.grid(axis="x", visible=False)

    ax2.axhline(0, color=INK2, linewidth=0.9)
    for s, col, mk, off, lab in ((c, C_COL, "D", -0.1, "C: CRT"), (cs, CS_COL, "o", 0.1, "Cs: scale-free CRT")):
        ax2.errorbar(x + off, s.d_fpr_pp, yerr=[s.d_fpr_pp - s.ci_lo, s.ci_hi - s.d_fpr_pp],
                     fmt=mk, color=col, markersize=5, markeredgecolor="white", markeredgewidth=0.6,
                     elinewidth=1.2, capsize=2, label=lab)
    ax2.set_ylabel("FPR change vs A (pp)")
    ax2.set_title("Lending Club, target FNR 10%")
    ax2.set_xticks(x, [f"{r:.0%}" for r in RATIOS])
    ax2.set_xlabel("Positive-class ratio")
    ax2.legend(loc="upper right", frameon=False)
    ax2.grid(axis="x", visible=False)

    fig.tight_layout(h_pad=0.8)
    save(fig, "fig4_diagnosis_fix")


if __name__ == "__main__":
    main()