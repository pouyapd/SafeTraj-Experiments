# SafeTraj-Experiments

**Trajectory-Level Evaluation of Neural Motion Prediction for Autonomous Wheelchair Navigation**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Project: REXASI-PRO](https://img.shields.io/badge/EU%20Project-REXASI--PRO-blue)](https://rexasi-pro.spindoxlabs.com/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pouyapd/SafeTraj-Experiments/blob/main/notebooks/demo_analysis.ipynb)

---

**Author:** Pouya Bathaei Pourmand  
**Affiliation:** MSc in Computer Engineering (AI) — University of Genoa / CNR, Italy  
**Project:** [REXASI-PRO](https://rexasi-pro.spindoxlabs.com/) — Reliable & Explainable AI for Smart Mobility (EU Horizon Europe)  
**Supervisors:** Prof. Luca Oneto · Dr. Maurizio Mongelli · Dr. Sara Narteni

---

## Overview

This repository contains the public results and source code from the MSc thesis:

> **"Analysis of Neural Trajectory Prediction for Collision Avoidance in Smart Wheelchairs"**

The thesis develops a **systematic trajectory-level evaluation framework** for pretrained neural motion prediction models (DNN-LNA) used in autonomous wheelchair navigation.

The core question is: **when and why do neural trajectory predictors fail?**

All five models are treated as **black-box predictors** — evaluated purely through their outputs, without any access to weights or retraining.

> ⚠️ The DNN-LNA model weights are **not included** (proprietary, REXASI-PRO project). Only derived results, CSVs and visualisations are public.

---

## Demo Notebook

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pouyapd/SafeTraj-Experiments/blob/main/notebooks/demo_analysis.ipynb)

A reproducible demo is available in [`notebooks/demo_analysis.ipynb`](notebooks/demo_analysis.ipynb).

It reproduces the key findings using **synthetic data** that matches the operational ranges of the REXASI-PRO wheelchair platform — no proprietary model weights or confidential datasets required.

**Covers:**
- Exp1 — Input sensitivity analysis and risk maps
- Exp2 — Goal difficulty analysis and model comparison
- XAI supervisor — Decision Tree and Random Forest on synthetic data
- Summary of all key findings

> **Note:** The notebook uses synthetic data to illustrate methodology and reproduce statistical patterns.
> Accuracy values reflect performance on synthetic data only.
> Real experimental results are in `results/figures_exp1/`, `results/figures_exp2/`, and `results/tables/`.

---

## Problem Setting

Each navigation scenario is defined by three input commands:

| Symbol | Variable | Operational Range |
|--------|----------|-------------------|
| φ (theta) | Initial orientation | [−π, π] rad |
| v | Linear velocity | [−1.05, 2.88] m/s |
| ω (omega) | Angular velocity | [−1.99, 1.99] rad/s |

Given these inputs, a pretrained neural model predicts a future trajectory:

```
X̂ = f_θ(φ, v, ω) = [(x₁,y₁,θ₁), ..., (x_T, y_T, θ_T)]    T = 30 steps
```

A trajectory is evaluated using two criteria:

- **Strict success:** final distance to goal `d < 0.30 m`
- **Soft success:** Success (`d < 0.30 m`) / Near-success (`0.30 ≤ d < 0.50 m`) / Failure (`d ≥ 0.50 m`)

---

## Model Performance Summary

Strict success rate across all five models and three goal configurations (N=200 × 3 goals = 600 observations per model):

![Strict Success Rate](results/figures/table_success_rate.png)

---

## Experiments

### Experiment 1 — Input-Space Sensitivity Analysis

**Objective:** Identify which command regions trigger failed trajectory predictions.

**Method:** N=200 inputs sampled uniformly across the operational range (φ, v, ω). Each of the five models generates a predicted trajectory per input. Outcomes are labelled as Success (`d < 0.30 m`) or Failure (`d ≥ 0.30 m`) and aggregated into risk maps over the input space.

**Key findings:**
- Initial orientation φ near ±π (wheelchair facing backward) is the dominant failure trigger
- Angular velocity ω has secondary influence
- Clear *zones of exclusion* exist in the command space

**3D Failure Zone Map:**

![3D Failure Zone Map](results/figures_exp1/3d_theta_vel_dist_DNN_LNA_on_wheel1_goal_1.0_0.0.png)

**Initial orientation vs. final distance to goal:**

![Theta vs Distance](results/figures_exp1/exp1_theta_vs_distance.png)

---

### Experiment 2 — Goal-Based Difficulty Analysis

**Objective:** Evaluate how model performance changes across different target positions.

**Method:** Three reference goals tested across all five models with strict and soft success criteria.

| Goal | Position | Complexity |
|------|----------|------------|
| A | (0.5, 0.5) | Near lateral — low/medium |
| B | (1.0, 0.0) | Straight ahead — medium |
| C | (1.5, −0.5) | Far off-axis — high |

**Key findings:**
- `DNN_LNA_closs2` achieves **99.3%** strict success across all goals
- `DNN_LNA_closs1` achieves only **25.3%** strict success
- Goal difficulty varies significantly across the workspace

**Goal difficulty map:**

![Goal Difficulty Map](results/figures_exp2/exp2_goal_difficulty_map_avg.png)

---

## Interpretability — Decision Tree Analysis

A shallow decision tree (max depth 4) was fitted per model on the
(φ, v, ω, GoalX, GoalY) → Success/Failure labels, revealing human-readable
rules for when each model fails.

![Decision Tree closs2](results/figures/tree_DNN_LNA_closs2.png)

> Full decision trees for all five models are available in [`results/figures/`](results/figures/).

---

## Practical Implications for Collision Avoidance

The identified risk regions support three concrete run-time applications:

1. **Dynamic Command Filtering** — Commands in high-risk regions (φ near ±π) can be proactively redirected before execution.
2. **Hybrid Navigation Fallback** — When the target lies in a high-failure region, the system switches to a classical model-based planner.
3. **Predictive Failure Monitoring** — Drift into the near-success zone triggers haptic or visual feedback to the user.

---

## Repository Structure

```
SafeTraj-Experiments/
│
├── notebooks/
│   └── demo_analysis.ipynb   # Reproducible demo with synthetic data
│
├── src/
│   ├── config.py             # Central configuration (ranges, goals, thresholds)
│   ├── model_utils.py        # Model loading and inference interface
│   └── evaluate.py           # Main evaluation pipeline (Exp1 + Exp2)
│
├── results/
│   ├── figures/              # Model-level results (tables + decision trees)
│   ├── figures_exp1/         # Experiment 1 visualisations
│   ├── figures_exp2/         # Experiment 2 visualisations
│   └── tables/               # CSV outputs
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Reproducing the Results

> The DNN-LNA model weights are not publicly available. To reproduce:
> 1. Obtain model weights from the [REXASI-PRO project](https://rexasi-pro.spindoxlabs.com/)
> 2. Place each model under `models/<model_name>/DNN_LNA_model/`
> 3. Run the pipeline:

```bash
pip install -r requirements.txt
python src/evaluate.py --model_root /path/to/models --output_dir results/
```

For a reproducible demo without model weights, see the [demo notebook](notebooks/demo_analysis.ipynb).

---

## Related Projects

- **[SafeTraj-Prototype](https://github.com/pouyapd/SafeTraj-Prototype)** — Trajectory behaviour analysis toolkit with live Streamlit dashboard and REST API
- **[SafeNav-RL](https://github.com/pouyapd/SafeNav-RL)** — Safety-constrained RL navigation agent extending this analysis work

---

## Citation

```bibtex
@mastersthesis{bathaei2026safetraj,
  author  = {Bathaei Pourmand, Pouya},
  title   = {Analysis of Neural Trajectory Prediction for Collision Avoidance
             in Smart Wheelchairs},
  school  = {University of Genoa, DIBRIS},
  year    = {2026},
  note    = {MSc in Computer Engineering (AI), REXASI-PRO EU Project}
}
```

---

## License

MIT — see [LICENSE](LICENSE) for details.  
Model weights are proprietary to the REXASI-PRO project and are not covered by this license.
