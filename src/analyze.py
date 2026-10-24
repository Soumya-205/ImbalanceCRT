"""Figures, summary table and paired statistics for the paper.

Reads results/{fraud,lending_club}_crt_v2.csv (v2 protocol) and writes:
  results/stats_paired.csv            every paired comparison (all LRs, both alphas)
  results/tables/summary_primary.tex  Table: primary protocol, alpha = 0.10
  results/figures/fig1_fpr_boxplots.(pdf|png)    FPR per model and ratio (primary)
  results/figures/fig2_paired_delta.(pdf|png)    paired FPR change vs A, 95% CI
  results/figures/fig3_lr_sensitivity.(pdf|png)  C - A at every fine-tuning LR

Statistics: per (dataset, alpha, ratio), the 10 seeds give 10 paired differences
(model - A, same split and same re-splits). We report the mean difference in
percentage points, a t-based 95% CI, the Wilcoxon signed-rank p-value, and a
Holm-adjusted p-value within each (dataset, alpha, comparison) family of 4 ratios.

Usage (from the project root):  python src/analyze.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # write files, no window
import matplotlib.pyplot as plt
from scipy.stats import t as t_dist, wilcoxon

from config import RESULTS_DIR, PRIMARY_FT_LR

DATASETS = {"fraud": "Credit Card Fraud", "lending_club": "Lending Club", "baf": "Bank Account Fraud"}
RATIOS = [0.01, 0.05, 0.10, 0.30]
# Colours: reference palette slots 1-3 (validated colourblind-safe as a set);
# baseline A is neutral grey. Hatching gives a second, print-safe cue.
STYLE = {
    "A":  dict(color="#8a8985", hatch="",     marker="o", label="A: BCE + CRC"),
    "B":  dict(color="#eb6834", hatch="//",   marker="s", label="B: weighted BCE + CRC"),
    "A+": dict(color="#1baf7a", hatch="..",   marker="^", label="A+: A + extra BCE (control)"),
    "C":  dict(color="#2a78d6", hatch="\\\\", marker="D", label="C: CRT (Yeh et al.)"),
}
LR_STYLE = {1e-4: ("#86b6ef", "o"), 1e-3: ("#2a78d6", "s"), 1e-2: ("#104281", "D")}  # ordinal blue ramp
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42,
})


# ------------------------------------------------------------------ loading
def load():
    frames = []
    for ds in DATASETS:
        f = RESULTS_DIR / f"{ds}_crt_v2.csv"
        if f.exists():
            frames.append(pd.read_csv(f))
        else:
            print(f"(missing {f.name}; skipped)")
    return pd.concat(frames, ignore_index=True)


def primary(df):
    """A and B, plus A+ and C at the pre-specified fine-tuning LR."""
    return df[df.model.isin(["A", "B"]) | np.isclose(df.lr.fillna(-1), PRIMARY_FT_LR)]


# --------------------------------------------------------------- statistics
def holm(p):
    """Holm step-down adjustment for a small family of p-values."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj, running = np.empty_like(p), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def paired_stats(df):
    """Every (model at an LR) minus A, per dataset/alpha/ratio, over the paired seeds."""
    a = df[df.model == "A"].set_index(["dataset", "alpha", "ratio", "seed"])
    out = []
    others = df[df.model != "A"].copy()
    others["lr_key"] = others.lr.fillna(-1)
    for (ds, al, model, lr), g in others.groupby(["dataset", "alpha", "model", "lr_key"]):
        for ratio, gr in g.groupby("ratio"):
            gr = gr.set_index(["dataset", "alpha", "ratio", "seed"])
            d = (gr.fpr - a.fpr.reindex(gr.index)).dropna() * 100       # percentage points
            n = len(d)
            half = t_dist.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
            p = wilcoxon(d, zero_method="zsplit").pvalue if (d != 0).any() else 1.0
            dauc = (gr.auroc - a.auroc.reindex(gr.index)).mean()
            out.append(dict(dataset=ds, alpha=al, model=model, lr=(np.nan if lr == -1 else lr),
                            ratio=ratio, n=n, d_fpr_pp=d.mean(), ci_lo=d.mean() - half,
                            ci_hi=d.mean() + half, wins=int((d < 0).sum()), p=p, d_auroc=dauc))
    st = pd.DataFrame(out)
    st["p_holm"] = np.nan
    for _, idx in st.groupby(["dataset", "alpha", "model", st.lr.fillna(-1)]).groups.items():
        st.loc[idx, "p_holm"] = holm(st.loc[idx, "p"])
    return st.sort_values(["dataset", "alpha", "model", "lr", "ratio"]).reset_index(drop=True)


# ------------------------------------------------------------------ figures
def save(fig, name):
    out = RESULTS_DIR / "figures"
    out.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)
    print(f"wrote figures/{name}.pdf/.png")


def fig1_boxplots(df, alpha=0.10):
    """FPR per model per ratio (primary protocol). Mirrors Yeh et al. Fig. 1."""
    d = primary(df)
    d = d[np.isclose(d.alpha, alpha)]
    models = ["A", "B", "A+", "C"]
    fig, axes = plt.subplots(len(DATASETS), 1, figsize=(3.5, 2.15 * len(DATASETS)))      # one IEEE column
    for ax, (ds, title) in zip(np.atleast_1d(axes), DATASETS.items()):
        sub = d[d.dataset == ds]
        width = 0.18
        for j, m in enumerate(models):
            data = [sub[(sub.ratio == r) & (sub.model == m)].fpr.values * 100 for r in RATIOS]
            pos = np.arange(len(RATIOS)) + (j - 1.5) * (width + 0.02)
            bp = ax.boxplot(data, positions=pos, widths=width, patch_artist=True,
                            showfliers=True, medianprops=dict(color=INK, linewidth=1),
                            whiskerprops=dict(color=INK2, linewidth=0.8),
                            capprops=dict(color=INK2, linewidth=0.8),
                            flierprops=dict(marker="o", markersize=2.5, markerfacecolor=STYLE[m]["color"],
                                            markeredgecolor="none"))
            for box in bp["boxes"]:
                box.set(facecolor=STYLE[m]["color"], alpha=0.85, edgecolor="white",
                        linewidth=0.8, hatch=STYLE[m]["hatch"])
        ax.set_xticks(range(len(RATIOS)), [f"{r:.0%}" for r in RATIOS])
        ax.set_xlabel("Positive-class ratio")
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    for ax in np.atleast_1d(axes):
        ax.set_ylabel(f"Test FPR (%)")
    for ax in np.atleast_1d(axes)[:-1]:
        ax.set_xlabel("")                       # x label only under the bottom panel
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=STYLE[m]["color"], hatch=STYLE[m]["hatch"],
                             edgecolor="white") for m in models]
    fig.legend(handles, [STYLE[m]["label"] for m in models], loc="upper center", ncol=2,
               frameon=False, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout(h_pad=0.6)
    save(fig, "fig1_fpr_boxplots")


def fig2_paired(st):
    """Paired FPR change vs A (mean, 95% CI), primary protocol, both alphas."""
    alphas = [0.10, 0.05]
    fig, axes = plt.subplots(len(DATASETS), len(alphas), figsize=(7.16, 1.6 * len(DATASETS)), sharex=True)
    models = ["B", "A+", "C"]
    for i, ds in enumerate(DATASETS):
        for k, al in enumerate(alphas):
            ax = axes[i, k]
            ax.axhline(0, color=INK2, linewidth=0.9)
            for j, m in enumerate(models):
                s = st[(st.dataset == ds) & np.isclose(st.alpha, al) & (st.model == m)]
                if m != "B":
                    s = s[np.isclose(s.lr, PRIMARY_FT_LR)]
                x = np.arange(len(RATIOS)) + (j - 1) * 0.2
                s = s.set_index("ratio").reindex(RATIOS)
                ax.errorbar(x, s.d_fpr_pp, yerr=[s.d_fpr_pp - s.ci_lo, s.ci_hi - s.d_fpr_pp],
                            fmt=STYLE[m]["marker"], color=STYLE[m]["color"], markersize=5,
                            markeredgecolor="white", markeredgewidth=0.6, elinewidth=1.2,
                            capsize=2, label=STYLE[m]["label"])
            ax.set_title(f"{DATASETS[ds]}, target FNR {al:.0%}" + (" (primary)" if al == 0.10 else ""))
            ax.set_xticks(range(len(RATIOS)), [f"{r:.0%}" for r in RATIOS])
            ax.grid(axis="x", visible=False)
            if i == len(DATASETS) - 1:
                ax.set_xlabel("Positive-class ratio")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.supylabel("FPR change vs A (pp)", fontsize=8, x=0.005)   # one shared y label
    fig.tight_layout(h_pad=0.8)
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.05))
    save(fig, "fig2_paired_delta")


def fig3_lr(st, alpha=0.10):
    """C - A at every fine-tuning LR (sensitivity)."""
    fig, axes = plt.subplots(len(DATASETS), 1, figsize=(3.5, 1.95 * len(DATASETS)))      # one IEEE column
    for ax, ds in zip(np.atleast_1d(axes), DATASETS):
        ax.axhline(0, color=INK2, linewidth=0.9)
        for j, (lr, (col, mk)) in enumerate(LR_STYLE.items()):
            s = st[(st.dataset == ds) & np.isclose(st.alpha, alpha) & (st.model == "C")
                   & np.isclose(st.lr, lr)].set_index("ratio").reindex(RATIOS)
            x = np.arange(len(RATIOS)) + (j - 1) * 0.2
            ax.errorbar(x, s.d_fpr_pp, yerr=[s.d_fpr_pp - s.ci_lo, s.ci_hi - s.d_fpr_pp],
                        fmt=mk, color=col, markersize=5, markeredgecolor="white",
                        markeredgewidth=0.6, elinewidth=1.2, capsize=2,
                        label=f"lr = {lr:g}" + (" (primary)" if np.isclose(lr, PRIMARY_FT_LR) else ""))
        ax.set_title(DATASETS[ds])
        ax.set_xticks(range(len(RATIOS)), [f"{r:.0%}" for r in RATIOS])
        ax.set_xlabel("Positive-class ratio")
        ax.grid(axis="x", visible=False)
    for ax in np.atleast_1d(axes):
        ax.set_ylabel("C $-$ A FPR (pp)")
    for ax in np.atleast_1d(axes)[:-1]:
        ax.set_xlabel("")                       # x label only under the bottom panel
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.tight_layout(h_pad=0.6)
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.06),
               handletextpad=0.2, columnspacing=0.8)
    save(fig, "fig3_lr_sensitivity")


# -------------------------------------------------------------------- table
def latex_table(df, st, alpha=0.10):
    """Mean test FPR (%) per model, and the paired C - A change, primary protocol."""
    d = primary(df)
    d = d[np.isclose(d.alpha, alpha)]
    means = d.groupby(["dataset", "ratio", "model"]).fpr.mean().unstack() * 100
    fnr = d.groupby(["dataset", "ratio"]).fnr.mean() * 100
    c = st[np.isclose(st.alpha, alpha) & (st.model == "C") & np.isclose(st.lr, PRIMARY_FT_LR)]
    c = c.set_index(["dataset", "ratio"])
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{Mean test FPR (\%) at target FNR $\alpha={alpha:.2f}$ over 10 seeds "
        r"(primary protocol). $\Delta$: paired C$-$A change in percentage points with 95\% CI; "
        r"$p_H$: Holm-adjusted Wilcoxon p-value over the four ratios.}",
        r"\label{tab:primary}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{llrrrrrrr}",
        r"\hline",
        r"Dataset & Ratio & FNR & A & B & A+ & C & $\Delta$ (C$-$A) [95\% CI] & $p_H$ \\",
        r"\hline",
    ]
    for ds, name in DATASETS.items():
        if ds not in means.index.get_level_values(0):
            continue
        for k, r in enumerate(RATIOS):
            m = means.loc[(ds, r)]
            s = c.loc[(ds, r)]
            lines.append(
                f"{name if k == 0 else ''} & {r:.0%} & {fnr.loc[(ds, r)]:.1f} & "
                f"{m['A']:.1f} & {m['B']:.1f} & {m['A+']:.1f} & {m['C']:.1f} & "
                f"{s.d_fpr_pp:+.2f} [{s.ci_lo:+.2f}, {s.ci_hi:+.2f}] & {s.p_holm:.3f} \\\\"
                .replace("%", r"\%"))
        lines.append(r"\hline")
    lines += [r"\end{tabular}", r"\end{table*}"]
    out = RESULTS_DIR / "tables"
    out.mkdir(exist_ok=True)
    (out / "summary_primary.tex").write_text("\n".join(lines))
    print("wrote tables/summary_primary.tex")


def main():
    df = load()
    st = paired_stats(df)
    st.to_csv(RESULTS_DIR / "stats_paired.csv", index=False)
    print(f"wrote stats_paired.csv ({len(st)} comparisons)")
    fig1_boxplots(df)
    fig2_paired(st)
    fig3_lr(st)
    latex_table(df, st)
    view = st[np.isclose(st.alpha, 0.10) & ((st.model == "B") | np.isclose(st.lr.fillna(-1), PRIMARY_FT_LR))]
    print("\nPrimary protocol, alpha = 0.10 (FPR change vs A, percentage points):")
    print(view[["dataset", "model", "ratio", "d_fpr_pp", "ci_lo", "ci_hi", "wins", "p", "p_holm"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()