# ImbalanceCRT: Conformal Risk Training Under Class Imbalance

An extension and stress test of Conformal Risk Training (CRT; Yeh et al., NeurIPS 2025).
Question: at a matched target false-negative rate (FNR), does CRT's false-positive-rate (FPR)
advantage over post-hoc Conformal Risk Control (CRC) hold, shrink, or reverse as the
positive-class ratio drops from 30% to 1%?

## Models (same MLP, hidden 64-32, same training budget)
- **A:** BCE, then post-hoc CRC
- **B:** class-weighted BCE (trained from scratch), then post-hoc CRC
- **C:** CRT (Yeh et al., Algorithm 2, FNR case), then CRC on a held-out calibration set

## Setup
- Imbalance ratios: 1%, 5%, 10%, 30% (majority class subsampled; all positives kept)
- Target FNR (alpha): 5%, 10%
- Splits: stratified 50/30/20 train/calibration/test; 10 seeds
- Datasets: ULB Credit Card Fraud; Lending Club (in progress)

## Run
    python -m venv venv; .\venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    python src/data.py                 # check splits
    python src/baselines.py --quick    # smoke test
    python src/baselines.py            # Models A/B, full sweep -> results/

## Status
Models A/B implemented. Model C in progress.