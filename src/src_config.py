"""
config.py — Central configuration for SafeTraj-Experiments
"""

import numpy as np

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42

# ── Model names ──────────────────────────────────────────────────────────────
MODEL_NAMES = [
    "DNN_LNA_closs1",
    "DNN_LNA_closs2",
    "DNN_LNA_mse",
    "DNN_LNA_on_wheel1",
    "DNN_LNA_sinu",
]

# ── Operational input ranges ─────────────────────────────────────────────────
INPUT_RANGES = {
    "theta": (-np.pi, np.pi),      # initial orientation [rad]
    "v":     (-1.05,  2.88),       # linear velocity [m/s]
    "omega": (-1.99,  1.99),       # angular velocity [rad/s]
}

# ── Experiment parameters ─────────────────────────────────────────────────────
NUM_SAMPLES_EXP1 = 200             # random inputs for Exp1
BATCH_SIZE       = 50              # inference batch size
MAP_SHAPE        = (300, 300, 1)   # dummy occupancy map shape

# ── Goal configurations (Exp2) ────────────────────────────────────────────────
GOALS = [
    (0.5,  0.5),   # Goal A — near lateral
    (1.0,  0.0),   # Goal B — straight ahead
    (1.5, -0.5),   # Goal C — far off-axis
]

# ── Success thresholds ────────────────────────────────────────────────────────
THRESHOLDS   = [0.1, 0.2, 0.3, 0.5, 1.0]
STRICT_THRESH = 0.30    # strict success: d < 0.30 m
NEAR_THRESH   = 0.50    # soft success: near-success zone boundary

# ── Output paths (relative to repo root) ─────────────────────────────────────
RESULTS_DIR   = "results/tables"
FIG_EXP1_DIR  = "results/figures_exp1"
FIG_EXP2_DIR  = "results/figures_exp2"
