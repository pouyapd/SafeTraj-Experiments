# SafeTraj-Experiments

**Trajectory-Level Evaluation of Neural Motion Prediction for Autonomous Wheelchair Navigation**

---

**Author:** Pouya Bathaei Pourmand  
**Affiliation:** MSc in Computer Engineering (AI) — University of Genoa / CNR-IEIIT, Italy  
**Project:** [REXASI-PRO](https://rexasi-pro.spindoxlabs.com/) — Reliable & Explainable AI for Smart Mobility (EU Horizon Europe)  
**Supervisors:** Prof. Luca Oneto · Dr. Maurizio Mongelli · Dr. Sara Narteni

---

## Overview

This repository contains the public results and visualizations from my MSc thesis:

> *"Analysis of Neural Trajectory Prediction for Collision Avoidance in Smart Wheelchairs"*

The thesis develops a systematic **trajectory-level evaluation framework** for pretrained neural motion prediction models (DNN-LNA) used in autonomous wheelchair navigation.

The focus is on understanding **when and why** neural predictors generate unstable or unsafe trajectories — without modifying or retraining the models. They are treated as black-box predictors and evaluated purely through their outputs.

The neural models analysed were developed and trained by project partners within the REXASI-PRO project. No proprietary model weights or confidential datasets are included in this repository.

---

## Problem Setting

Each navigation scenario is defined by three input commands:

| Input | Symbol | Range |
|---|---|---|
| Initial orientation | φ | [−π, π] rad |
| Linear velocity | v | [−1.05, 2.88] m/s |
| Angular velocity | ω | [−1.99, 1.99] rad/s |

Given these inputs, a pretrained neural model predicts a future trajectory:

```
X̂ = f_θ(φ, v, ω) = [x̂₁, x̂₂, ..., x̂_T]
```

A trajectory is evaluated using two success criteria:
- **Strict success:** final distance to goal d < 0.30 m
- **Soft success:** distinguishes Success (d < 0.30 m), Near-success (0.30 ≤ d < 0.50 m), and Failure (d ≥ 0.50 m)

---

## Experiments

### Experiment 1 — Input-Space Sensitivity Analysis

**Objective:** Identify which input command regions trigger unstable or unsafe trajectory predictions.

**Method:** 200 inputs sampled uniformly across the operational range (φ, v, ω). For each input, all five DNN-LNA models generate a predicted trajectory. Trajectories are evaluated and aggregated into risk maps over the input space.

**Key findings:**
- Initial orientation φ is the **dominant risk factor** — failures concentrate near ±π (robot facing away from goal)
- Angular velocity ω has secondary influence on instability
- Risk is not uniformly distributed — clear "zones of exclusion" exist in the command space
- DNN-LNA-closs1 is the most unstable; DNN-LNA-closs2 is the most stable

**Outputs:**
- φ / v / ω vs. distance-to-goal plots
- Risk maps over (φ, ω) space
- Failure-distance boxplots per model

#### Orientation vs. Distance to Goal
![Theta vs Distance](https://raw.githubusercontent.com/pouyapd/SafeTraj-Experiments/main/results/figures_exp1/exp1_theta_vs_distance.png)

---

### Experiment 2 — Goal-Based Difficulty Analysis

**Objective:** Evaluate how model performance changes across different target positions and identify which goals are inherently harder for neural predictors.

**Method:** Three reference goal configurations tested across all models, evaluated with both strict and soft success criteria.

**Key findings:**
- Goal reachability is strongly goal-dependent
- DNN-LNA-closs2 achieves **99.3% strict success** across all goals
- DNN-LNA-closs1 achieves only **25.3% strict success**
- Soft success analysis reveals near-miss cases invisible to binary metrics
- High-curvature goals requiring sharp maneuvers expose the largest model differences

**Outputs:**
- Goal difficulty maps per model
- Soft outcome distribution (Success / Near-success / Failure)
- Strict success rate table across all 5 models

#### Goal Difficulty Map
![Goal Difficulty Map](https://raw.githubusercontent.com/pouyapd/SafeTraj-Experiments/main/results/figures_exp2/exp2_goal_difficulty_map_avg.png)

---

## Models Evaluated

Five pretrained DNN-LNA (Deep Neural Network — Local Navigation Approach) models:

| Model | Strict Success (Exp2) |
|---|---|
| DNN-LNA-closs1 | 25.3% |
| DNN-LNA-closs2 | 99.3% |
| DNN-LNA-mse | 74.7% |
| DNN-LNA-on-wheel1 | 90.7% |
| DNN-LNA-sinu | 69.2% |

---

## Practical Implications

The risk maps and goal difficulty patterns derived from this work support three concrete run-time safety applications:

**1. Dynamic Command Filtering**
Commands falling into identified high-risk regions (φ near ±π) can be proactively scaled down or redirected before execution.

**2. Hybrid Navigation Fallback**
When the current goal lies in a high-difficulty region, the system can automatically switch from the neural predictor to a classical model-based planner.

**3. Predictive Failure Monitoring**
Consistent drift into Near-success regions can trigger haptic or visual feedback to the user, indicating reduced reliability.

---

## Repository Contents

```
SafeTraj-Experiments/
├── results/
│   ├── tables/                         # CSV outputs from Exp1 & Exp2
│   │   ├── summary_avg_success.csv
│   │   ├── combined_overview.csv
│   │   ├── exp1_theta_sensitivity.csv
│   │   ├── exp1_failure_distance_summary.csv
│   │   ├── exp2_goal_summary_per_model.csv
│   │   └── exp2_goal_mean_over_models.csv
│   │
│   ├── figures_exp1/                   # Experiment 1 visualizations
│   │   ├── exp1_theta_vs_distance.png
│   │   ├── exp1_velocity_vs_distance.png
│   │   ├── exp1_omega_vs_distance.png
│   │   ├── exp1_failure_distance_boxplot.png
│   │   └── exp1_goal_theta_success_3d.png
│   │
│   └── figures_exp2/                   # Experiment 2 visualizations
│       ├── exp2_goal_difficulty_map_avg.png
│       └── exp2_success_heatmap.png
```

---

## Disclaimer

This repository contains only derived results, aggregated summaries, and public-safe visualizations.  
No confidential datasets, proprietary model weights, or industrial source code are included.  
The DNN-LNA models were developed by project partners within the REXASI-PRO framework.

---

## Related Projects

- [SafeTraj-Prototype](https://github.com/pouyapd/SafeTraj-Prototype) — Python toolkit for trajectory behaviour analysis and risk scoring, built alongside this research
- [SafeNav-RL](https://github.com/pouyapd/SafeNav-RL) — RL-based navigation agent extending this evaluation work toward safety-constrained policy learning

---

*Part of MSc thesis research at the University of Genoa, in collaboration with CNR-IEIIT within the EU Horizon Europe REXASI-PRO project.*
