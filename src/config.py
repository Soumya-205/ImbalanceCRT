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