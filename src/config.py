"""Shared configuration for ImbalanceCRT experiments."""

from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# --- Experiment grid ---
IMBALANCE_RATIOS = [0.01, 0.05, 0.10, 0.30]   # fraction of positives (fraud) per sweep level
TARGET_FNR_LEVELS = [0.01, 0.05, 0.10]        # alpha: desired false-negative-rate guarantee
RANDOM_SEEDS = list(range(10))                # matches the anchor paper's 10-seed protocol

# --- Split proportions (within each imbalance level) ---
TRAIN_FRAC = 0.6
CAL_FRAC = 0.2
TEST_FRAC = 0.2