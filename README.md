# Safe & Explainable AI for Trajectory Prediction 
This README has been updated in 2025 to include the new experiments (Exp1 and Exp2) and failure-distance analysis

**Author:** Pouya Bathaie Pourmand  
**Affiliation:** MSc in Artificial Intelligence — University of Genoa / CNR Italy  
**Project:** REXASI-PRO — Reliable & Explainable AI for Smart Mobility  
**Focus:** Understanding and supervising neural trajectory-prediction models to ensure safety and robustness in autonomous mobility systems.

---

![3D Safety Structure](results/figures_exp1/exp1_goal_theta_success_3d.png)
*3D view of input–goal interactions showing success/failure regions in the model’s behavior.*

---

## 1. Overview
This repository contains a public summary of my MSc research on **safe and explainable AI for trajectory prediction**, with applications to smart wheelchairs and mobile robots.

The goal is to:
- analyse **when and why** neural predictors fail,  
- identify **unsafe input conditions**,  
- and design **interpretable supervisors** that assess risk and flag potentially unsafe behaviours before execution.

The neural models analysed here were **pre-trained by an industrial partner**.  
Only behaviour analysis and failure characterization are included — **no proprietary data or models** are shared.

---

## 2. Problem Setting
Each scenario is defined by three controllable inputs:

- **Orientation (θ)**
- **Linear velocity (v)**
- **Rotational velocity (ω)**

Given these parameters, a network predicts a **trajectory** toward a target goal.  
A prediction is labelled as:

- **Success** → ends sufficiently close to goal  
- **Failure** → deviates significantly or diverges

This allows evaluation of stability and unsafe behaviour modes.

---

## 3. Experiments (2025 Update)

### Experiment 1 — Input Sensitivity (Fixed Goal)
Purpose: Identify unsafe regions in the input space (θ, v, ω).

**Key findings:**
- **Orientation (θ)** is the strongest risk factor  
  → failures increase near ±π (robot facing backwards)
- High **v** or **ω** have secondary effects
- **closs1**: most unstable  
- **closs2**: most stable; low failure distance even when failing

**Included outputs:**
- θ / v / ω vs distance plots  
- Success-rate bins  
- Failure-distance boxplot  
- CSV summaries  

---

### Experiment 2 — Goal Difficulty (Fixed Start)
Purpose: Evaluate which goal positions are more difficult.

**Key findings:**
- Goals **(0.5, 0.5)** and **(1.5, –0.5)** → almost always successful  
- Goal **(1.0, 0.0)** → systematically difficult  
- Difficulty maps separate safe vs challenging regions

**Included outputs:**
- Success heatmap  
- Difficulty map  
- Summary tables  

---

### Failure-Distance Analysis
Beyond success/failure, the analysis measures **how severe** failures are.

- **closs1**: failures up to ~1 m  
- **closs2**: failures mostly under ~0.33 m  

This metric is essential for safety supervision.

---

## 4. Explainable AI Supervisor
Interpretable models trained using labels derived from neural predictions, used as post-hoc analysis tools to explain and summarise trajectory behaviour.

- **Decision Tree (depth=8)**  
- **Random Forest**  
- **KNN**

**Performance:**
- Random Forest → ~98% accuracy  
- Decision Tree → ~97.5% and provides human-readable rules  
- KNN → ~95%, slightly biased toward “Success”

**Feature importance:**
1. Orientation (θ)  
2. Linear velocity (v)  
3. Rotational velocity (ω)

TThese highlight risk-sensitive regions of the input space and support interpretability and system understanding.

---

## 5. Repository Contents

The project file structure is organized as follows:

```text
.
├── results/
│   ├── tables/                       # All CSV outputs from Exp1 & Exp2
│   │   ├── summary_avg_success.csv
│   │   ├── combined_overview.csv
│   │   ├── exp1_theta_sensitivity.csv
│   │   ├── exp1_input_sensitivity_goal_1_0.csv
│   │   ├── exp1_failure_distance_summary.csv
│   │   ├── exp2_goal_summary_per_model.csv
│   │   ├── exp2_goal_mean_over_models.csv
│   │   └── exp1_bad_theta_bins.csv
│   │
│   ├── figures_exp1/                 # Key visualizations from Experiment 1
│   │   ├── exp1_theta_vs_distance.png
│   │   ├── exp1_velocity_vs_distance.png
│   │   ├── exp1_omega_vs_distance.png
│   │   ├── exp1_failure_distance_boxplot.png
│   │   ├── exp1_theta_vs_goal_distance_success.png
│   │   └── exp1_goal_theta_success_3d.png
│   │
│   └── figures_exp2/                 # Key visualizations from Experiment 2
│       ├── exp2_goal_difficulty_map_avg.png
│       └── exp2_success_heatmap.png
│
└── figures/                          # Older/general project figures
```

## 6. Applications
This work supports:

- Safety monitoring in smart wheelchairs  
- Trajectory supervision in mobile robots  
- Navigation risk analysis  
- Model monitoring and behaviour-based risk analysis pipelines 

The core principles generalize to any **safety-critical AI system** where robustness and interpretability are required.

---

## 7. Disclaimer
This repository contains only **derived results, summaries, and public-safe visualizations**.  
No confidential datasets, model weights, or industrial source code are included.

