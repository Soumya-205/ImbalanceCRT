"""Shared configuration for ImbalanceCRT experiments."""

from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# --- Experiment grid ---
IMBALANCE_RATIOS = [0.01, 0.05, 0.10, 0.30]   # fraction of positives per sweep level
TARGET_FNR_LEVELS = [0.05, 0.10]              # alpha (0.01 dropped: infeasible at 148 cal positives)
RANDOM_SEEDS = list(range(10))

# --- Lending Club: number of defaults sampled per seed and kept at every ratio
#     (gives 750/450/300 train/cal/test positives with the 50/30/20 split) ---
LC_N_POS = 1500
# --- Split proportions (within each imbalance level) ---
TRAIN_FRAC = 0.5
CAL_FRAC = 0.3
TEST_FRAC = 0.2

# --- Training settings shared by Models A, B and C (same budget = fair comparison) ---
HIDDEN = (64, 32)
EPOCHS = 30
LR = 1e-3
BATCH_SIZE = 256

N_RESPLITS=100



# --- Model C (conformal risk training) and control A+; values from Yeh et al.'s
#     official code (run_polyp.py) unless marked ---
CRT_T = 0.1                     # sigmoid temperature in the smooth FPR loss (Yeh: T=0.1)
FT_LRS = [1e-2, 1e-3, 1e-4]     # fine-tuning LR grid (Yeh: 1e-2..1e-6; 1e-5/1e-6 dropped for CPU)
FT_MAX_EPOCHS = 100             # Yeh: MAX_EPOCHS = 100
FT_PATIENCE = 10                # Yeh: stop after >10 epochs without val improvement
FT_STEPS_PER_EPOCH = 20         # ours: fixed optimisation budget per epoch at every ratio
CRT_NEG_PER_STEP = 256          # ours: negatives per step for the FPR loss
VAL_FRAC = 0.2                  # ours: share of TRAIN held out for early stopping / LR choice

BASE_MAX_EPOCHS = 500           # A/B in train_crt.py: early stopping on val loss, at most this many epochs
PRIMARY_FT_LR = 1e-3            # pre-specified (Oct 15, before the v2 run) primary fine-tuning LR for C and A+