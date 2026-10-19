# ImbalanceCRT: Conformal Risk Training Under Class Imbalance

**A stress test of Conformal Risk Training (CRT; Yeh et al., NeurIPS 2025) on imbalanced tabular data.**

Conformal risk control (CRC) picks a decision threshold *after* training so that a model's
expected false-negative rate (FNR) stays below a target. CRT instead trains the model *through*
that threshold, and was reported to lower false positives on image segmentation. This project
asks whether that advantage carries over to imbalanced tabular binary classification:

> At a matched target FNR, does CRT's false-positive-rate (FPR) advantage over post-hoc CRC
> **hold, shrink, or reverse** as the positive-class ratio drops from 30% to 1%?

This is an extension and stress test of an existing method, not a new method. CRT is used as
specified in the [official implementation](https://github.com/chrisyeh96/conformal-risk-training),
with four documented protocol deviations (below).

---

## Key findings

At a target FNR of 10% (primary comparison), over 10 seeds × 100 calibration/test re-splits:

| Dataset | Positive ratio | FPR change, CRT − post-hoc CRC (95% CI) |
|---|---|---|
| Credit Card Fraud | 1% / 5% / 10% / 30% | +0.03 / −0.15 / +0.16 / +0.07 pp, **all CIs within ±0.53 pp** |
| Lending Club | 1% | **+4.09 pp** [2.62, 5.56] |
| Lending Club | 5% | **+3.84 pp** [2.03, 5.64] |
| Lending Club | 10% | **+2.17 pp** [0.98, 3.36] |
| Lending Club | 30% | +0.56 pp [−0.53, 1.64], not significant |

- **The advantage does not transfer.** CRT is neutral on credit card fraud and *increases* false
  positives on Lending Club, more so as positives become rarer.
- **Robust to the fine-tuning learning rate:** on Lending Club, CRT raised FPR at every rate tried
  (10⁻⁴, 10⁻³, 10⁻²) and under Yeh et al.'s validation-based selection.
- **Post-hoc CRC met the FNR target within sampling noise** for every method, and its threshold was
  close to the best achievable (0.3–1 pp above an oracle), so FPR was driven mainly by ranking quality.
- **Simple class weighting helped** on Lending Club at 1% and 5% positives (−1.8 and −2.1 pp).

![Paired change in FPR relative to model A](results/figures/fig2_paired_delta.png)

*Paired change in test FPR relative to model A (mean and 95% CI over 10 seeds). Below zero = fewer
false positives than A. Top: credit card fraud; bottom: Lending Club.*

<table>
<tr>
<td width="50%"><img src="results/figures/fig1_fpr_boxplots.png" alt="Test FPR per model and positive-class ratio"></td>
<td width="50%"><img src="results/figures/fig3_lr_sensitivity.png" alt="CRT minus A at each fine-tuning learning rate"></td>
</tr>
<tr>
<td><em>Test FPR at target FNR 10% for each model and positive-class ratio (10 seeds). On Lending Club,
CRT (blue) sits above the others at 1&ndash;10% positives.</em></td>
<td><em>FPR change of CRT relative to model A at each fine-tuning learning rate (target FNR 10%).
On Lending Club the increase grows with the learning rate and is smallest at 30% positives.</em></td>
</tr>
</table>

Full statistics for every comparison are in [`results/stats_paired.csv`](results/stats_paired.csv).

---

## What is compared

All models share one MLP (hidden layers 64 and 32, ReLU), trained with Adam (lr 10⁻³, batch 256).

| Model | Training | Threshold |
|---|---|---|
| **A** | Binary cross-entropy, early stopping on validation loss | Post-hoc CRC |
| **B** | Class-weighted cross-entropy (weight = n_neg / n_pos), trained from scratch, early stopping on validation loss | Post-hoc CRC |
| **C** | Model A fine-tuned with **CRT** (one model per target FNR) | Post-hoc CRC |
| **A+** | Model A fine-tuned with plain cross-entropy for the same budget (control) | Post-hoc CRC |

A+ exists so that any gain of C cannot be explained by extra training.

**Datasets**

| | Credit Card Fraud (ULB) | Lending Club |
|---|---|---|
| Source | kagglehub `mlg-ulb/creditcardfraud` | kagglehub `wordsforthewise/lending-club` |
| Rows used | 284,807 transactions | 1,345,350 resolved loans (Fully Paid vs Charged Off / Default) |
| Positives kept per ratio | all 492 frauds | 1,500 defaults sampled per seed |
| Features | 30 (28 PCA + time + amount) | 50 (only columns known at loan issue; no leakage) |
| Train / cal / test positives | 246 / 148 / 98 | 750 / 450 / 300 |

**Protocol**

- Imbalance sweep: negatives subsampled so positives are **30%, 10%, 5%, 1%** of the data.
- Splits: stratified 50 / 30 / 20 (train / calibration / test); 20% of train held out for validation.
- Target FNR α ∈ {0.05, 0.10}; **α = 0.10 is primary**.
- CRC is applied on logits; each trained model is evaluated on **100 random calibration/test
  re-splits**, identical across models for a given seed (paired).
- Statistics: paired differences vs A, t-based 95% CIs, Wilcoxon signed-rank tests,
  Holm correction over the four ratios.
- **Primary:** C and A+ fine-tuned at lr 10⁻³ (fixed before the final runs). **Sensitivity:**
  lr 10⁻⁴ and 10⁻², and Yeh-style selection by validation metric.

**Deviations from the official CRT protocol** (also stated in the paper and in `src/train_crt.py`)

1. Each CRT step uses a fixed number of positives (a random half of the training positives) as the
   pseudo-calibration set, plus 256 sampled negatives for the loss; random batches at 1% positives
   contain too few positives for CRC to be feasible.
2. Early stopping and learning-rate choice use a validation split from the training data, not the
   calibration set, so calibration and test data are never seen in training.
3. Learning-rate grid {10⁻⁴, 10⁻³, 10⁻²} instead of 10⁻⁶–10⁻², for CPU cost.
4. A fixed 20 optimization steps per fine-tuning epoch at every ratio.

---

## Repository layout

```
ImbalanceCRT/
├── src/
│   ├── config.py       # all settings: ratios, alphas, seeds, splits, training and CRT hyperparameters
│   ├── data.py         # download, cleaning, imbalance sweep, train-only imputation/scaling
│   ├── models.py       # the MLP and logit scoring
│   ├── crc.py          # post-hoc CRC for FNR, FNR/FPR metrics, oracle FPR
│   ├── baselines.py    # Model A/B training and the shared re-split evaluation
│   ├── train_crt.py    # main experiment: A, B, A+ and C (CRT) for one dataset
│   └── analyze.py      # paired statistics, LaTeX table and figures
├── results/
│   ├── fraud_crt_v2.csv          # main results, credit card fraud
│   ├── lending_club_crt_v2.csv   # main results, Lending Club
│   ├── stats_paired.csv          # every paired comparison vs A
│   ├── tables/summary_primary.tex
│   ├── figures/                  # fig1-fig3 as PDF and PNG
│   └── fraud_baselines*.csv, fraud_crt.csv   # earlier runs, kept for the record (see below)
├── requirements.txt
└── README.md
```

---

## Reproducing the results

Everything runs on a CPU; no GPU is needed. Tested with Python 3.12 on Windows.

**1. Set up**

```bash
git clone https://github.com/Soumya-205/ImbalanceCRT.git
cd ImbalanceCRT
python -m venv venv
# Windows (PowerShell):  .\venv\Scripts\Activate.ps1
# macOS / Linux:         source venv/bin/activate
pip install -r requirements.txt
```

**2. Check the data pipeline** (downloads the datasets through `kagglehub` on first run;
Lending Club is about 1.3 GB and is cached as `data/lending_club_clean.pkl` after cleaning)

```bash
python src/data.py
python src/data.py --dataset lending_club
```

**3. Run the experiments** (each sweep: 4 ratios × 10 seeds; Lending Club takes longest)

```bash
python src/train_crt.py --quick                          # smoke test: fraud, 1 seed, 5% ratio
python src/train_crt.py                                  # -> results/fraud_crt_v2.csv
python src/train_crt.py --dataset lending_club           # -> results/lending_club_crt_v2.csv
```

**4. Produce the statistics, table and figures**

```bash
python src/analyze.py
```

This writes `results/stats_paired.csv`, `results/tables/summary_primary.tex` and
`results/figures/fig1_fpr_boxplots`, `fig2_paired_delta`, `fig3_lr_sensitivity` (PDF and PNG).
Running it on the CSVs in this repository reproduces the numbers above exactly.

All settings live in `src/config.py`.

---

## Earlier runs

The repository also keeps earlier fraud-only results for transparency:

- `fraud_baselines_single_split.csv`, `fraud_baselines.csv`: Models A and B before the re-split
  evaluation and early stopping were introduced (`src/baselines.py`).
- `fraud_crt.csv` (v1): the first CRT run. It selected the fine-tuning learning rate on a small
  validation set (which mostly chose 10⁻², the worst rate on test data) and trained A for a fixed
  30 epochs (only 90 gradient steps at the 30% level). The protocol was revised to fix both
  issues before the final v2 runs; v1 is not used for conclusions.

---

## Limitations

- One architecture (a small MLP) and two datasets.
- Positives are fixed and negatives vary, so imbalance is confounded with training-set size;
  comparisons between models at a given ratio are fair, but trends across ratios partly reflect size.
- Few positives on fraud (98 test, 148 calibration), so α = 0.05 comparisons have low power.
- Fraud duplicates are retained; Lending Club uses resolved loans only; splits are random, not temporal.
- The CRT temperature (T = 0.1, from the official code) was not tuned.

---

## References

- C. Yeh, N. Christianson, A. Wierman, Y. Yue. *Conformal Risk Training: End-to-End Optimization of
  Conformal Risk Control.* NeurIPS 2025. [arXiv:2510.08748](https://arxiv.org/abs/2510.08748) ·
  [code](https://github.com/chrisyeh96/conformal-risk-training)
- A. N. Angelopoulos, S. Bates, A. Fisch, L. Lei, T. Schuster. *Conformal Risk Control.* ICLR 2024.
  [arXiv:2208.02814](https://arxiv.org/abs/2208.02814)
- A. Dal Pozzolo, O. Caelen, R. A. Johnson, G. Bontempi. *Calibrating Probability with
  Undersampling for Unbalanced Classification.* IEEE SSCI 2015 (credit card fraud dataset).

## Citation

If you use this code or results, please cite:

```bibtex
@misc{shradha2026imbalancecrt,
  author = {Shradha, Soumya},
  title  = {Conformal Risk Training Under Class Imbalance: A Stress Test on Tabular Data},
  year   = {2026},
  howpublished = {\url{https://github.com/Soumya-205/ImbalanceCRT}}
}
```

## Author

**Soumya Shradha**, Department of Computer Science and Engineering, Manipal University Jaipur ·
[GitHub](https://github.com/Soumya-205)