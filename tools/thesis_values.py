"""
Values reported in the MSc thesis (Tables 4.1 and 4.2, decision-tree summary),
exactly as quoted in README.md. Used by verify_results.py to check that the
published result files are consistent with them. This file only restates
reported numbers; it does not change or recompute them.
"""

MODELS = ["DNN_LNA_closs1", "DNN_LNA_closs2", "DNN_LNA_mse", "DNN_LNA_on_wheel1", "DNN_LNA_sinu"]
GOALS = [(0.5, 0.5), (1.0, 0.0), (1.5, -0.5)]          # A, B, C
N_INPUTS = 200                                          # sampled commands per model-goal cell

# Table 4.1 — strict success rate (%) per goal A, B, C and overall
SUCCESS_RATE = {
    "DNN_LNA_closs1": (50.5, 10.5, 15.0, 25.3),
    "DNN_LNA_closs2": (100.0, 98.0, 100.0, 99.3),
    "DNN_LNA_mse": (83.5, 76.5, 64.0, 74.7),
    "DNN_LNA_on_wheel1": (91.0, 88.0, 93.0, 90.7),
    "DNN_LNA_sinu": (79.5, 64.5, 63.5, 69.2),
}

# Table 4.2 — average final distance to goal (m) per goal A, B, C and overall
AVG_DISTANCE = {
    "DNN_LNA_closs1": (0.305, 0.603, 0.538, 0.482),
    "DNN_LNA_closs2": (0.123, 0.137, 0.129, 0.130),
    "DNN_LNA_mse": (0.200, 0.228, 0.283, 0.237),
    "DNN_LNA_on_wheel1": (0.188, 0.190, 0.150, 0.176),
    "DNN_LNA_sinu": (0.216, 0.261, 0.255, 0.244),
}

# Decision-tree summary — strict Success / Failure counts out of 600 observations
TREE_CLASS_COUNTS = {
    "DNN_LNA_closs2": (596, 4),
    "DNN_LNA_on_wheel1": (544, 56),
    "DNN_LNA_mse": (448, 152),
    "DNN_LNA_sinu": (415, 185),
    "DNN_LNA_closs1": (152, 448),
}
