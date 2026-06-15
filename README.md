# SafeTraj-Experiments

**Trajectory-Level Analysis of Neural Trajectory Prediction Models for Smart Wheelchair Navigation**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Project: REXASI-PRO](https://img.shields.io/badge/EU%20Project-REXASI--PRO-blue)](https://rexasi-pro.spindoxlabs.com/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pouyapd/SafeTraj-Experiments/blob/main/notebooks/demo_analysis.ipynb)

---

**Author:** Pouya Bathaei Pourmand
**Affiliation:** MSc in Computer Engineering (AI) — University of Genoa / CNR, Italy
**Project:** REXASI-PRO — Reliable & Explainable AI for Smart Mobility (EU Horizon Europe)
**Supervisors:** Prof. Luca Oneto · Prof. Maurizio Mongelli · Dr. Sara Narteni

---

## Overview

This repository contains the public results and source code from the MSc thesis:

> **Analysis of Neural Trajectory Prediction for Collision Avoidance in Smart Wheelchairs**

The thesis develops a systematic trajectory-level evaluation framework for pretrained neural motion prediction models (DNN-LNA) used in autonomous wheelchair navigation.

The central research question is:

> **When and why do neural trajectory predictors fail?**

All five models are treated as black-box predictors and are evaluated exclusively through their outputs, without access to internal weights, training procedures, or retraining.

> ⚠️ The original DNN-LNA model weights are not included because they are proprietary assets of the REXASI-PRO project. Only derived results, figures, tables and evaluation code are publicly available.

---

## Key Contributions

* Systematic trajectory-level evaluation of five DNN-LNA models
* Input-space risk mapping using sampled navigation commands
* Goal-based performance analysis using strict and soft success metrics
* Explainability through Decision Tree models
* Identification of critical failure regions for smart wheelchair navigation
* Comparative analysis of model robustness under different operating conditions

---

## Demo Notebook

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pouyapd/SafeTraj-Experiments/blob/main/notebooks/demo_analysis.ipynb)

A reproducible demonstration is available in:

```text
notebooks/demo_analysis.ipynb
```

The notebook reproduces the main methodology using synthetic data that follows the operational ranges of the REXASI-PRO wheelchair platform. No proprietary model weights or confidential datasets are required.

### Contents

* Experiment 1 — Input-space sensitivity analysis
* Experiment 2 — Goal-based performance analysis
* Decision Tree explainability
* Risk maps and failure region visualization
* Summary of key findings

> Note: The notebook uses synthetic data exclusively for demonstration purposes. Real experimental results are provided in the repository under the `results/` directory.

---

## Problem Setting

Each navigation scenario is defined by three motion commands:

| Symbol | Variable            | Operational Range   |
| ------ | ------------------- | ------------------- |
| θ      | Initial orientation | [−π, π] rad         |
| v      | Linear velocity     | [−1.05, 2.88] m/s   |
| ω      | Angular velocity    | [−1.99, 1.99] rad/s |

Together with a target goal position, these variables are provided to a pretrained DNN-LNA model, which predicts a future trajectory:

```text
X̂ = f(θ, v, ω, GoalX, GoalY)
```

The output trajectory consists of a sequence of future wheelchair states over a prediction horizon of 30 time steps.

Trajectory quality is evaluated using two criteria:

### Strict Success

```text
d < 0.30 m
```

### Soft Success

* Success: d < 0.30 m
* Near-success: 0.30 ≤ d < 0.50 m
* Failure: d ≥ 0.50 m

where d represents the final distance between the predicted trajectory endpoint and the target goal.

---

## Model Performance Summary

Strict success rate across all five models and three goal configurations:

```text
N = 200 samples × 3 goals = 600 evaluations per model
```

![Strict Success Rate](results/figures/table_success_rate.png)

---

## Experiments

### Experiment 1 — Input-Space Sensitivity Analysis

#### Objective

Identify which command regions are associated with failed trajectory predictions.

#### Method

A total of 200 input combinations were uniformly sampled across the operational ranges of:

* Initial orientation θ
* Linear velocity v
* Angular velocity ω

Each sampled configuration was evaluated by all five pretrained DNN-LNA models.

For each prediction, the final distance between the predicted trajectory endpoint and the target goal was computed and used to assign a Success or Failure label.

The resulting outcomes were aggregated into risk maps to identify critical operating regions.

#### Main Findings

* Initial orientation θ near ±π is the dominant failure trigger
* Angular velocity ω has a secondary influence
* Distinct failure regions emerge in the command space
* Model robustness varies significantly across operating conditions

#### Example Visualization

**3D Failure Zone Map**

![3D Failure Zone Map](results/figures_exp1/3d_theta_vel_dist_DNN_LNA_on_wheel1_goal_1.0_0.0.png)

**Orientation vs Final Distance**

![Theta vs Distance](results/figures_exp1/exp1_theta_vs_distance.png)

---

### Experiment 2 — Goal-Based Performance Analysis

#### Objective

Evaluate model behavior under different target positions.

#### Method

Three representative goals were tested using both strict and soft success criteria.

| Goal | Position    | Relative Difficulty |
| ---- | ----------- | ------------------- |
| A    | (0.5, 0.5)  | Low–Medium          |
| B    | (1.0, 0.0)  | Medium              |
| C    | (1.5, -0.5) | Medium–High         |

#### Main Findings

* DNN_LNA_closs2 achieved the highest overall performance
* DNN_LNA_closs1 showed the weakest performance
* Goal position affects performance, although its influence is smaller than orientation and velocity
* Some trajectories classified as failures under the strict metric become near-successes under the soft metric

#### Goal Success Patterns

![Goal Success Map](results/figures_exp2/exp2_goal_difficulty_map_avg.png)

---

## Explainability Through Decision Trees

To improve interpretability, a shallow Decision Tree was trained for each model using:

```text
(θ, v, ω, GoalX, GoalY)
```

as inputs and:

```text
Success / Failure
```

as labels.

The extracted trees provide explicit and human-readable rules that describe model behavior and identify critical failure regions.

### Example

![Decision Tree closs2](results/figures/tree_DNN_LNA_closs2.png)

Additional trees for all evaluated models are available in:

```text
results/figures/
```

---

## Practical Implications

The identified risk regions suggest several potential runtime applications:

### Dynamic Command Filtering

Commands located in high-risk regions can be modified before execution.

### Hybrid Navigation Fallback

The navigation system can switch to a classical planner when the predicted risk becomes excessive.

### Predictive Failure Monitoring

Near-success conditions can be detected early and used to trigger user feedback or additional safety mechanisms.

---

## Repository Structure

```text
SafeTraj-Experiments/
│
├── notebooks/
│   └── demo_analysis.ipynb
│
├── src/
│   ├── config.py
│   ├── model_utils.py
│   └── evaluate.py
│
├── results/
│   ├── figures/
│   ├── figures_exp1/
│   ├── figures_exp2/
│   └── tables/
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Reproducing the Results

The original DNN-LNA model weights are not publicly available.

To reproduce the experiments:

1. Obtain the pretrained models from the REXASI-PRO project
2. Place each model under:

```text
models/<model_name>/DNN_LNA_model/
```

3. Execute:

```bash
pip install -r requirements.txt
python src/evaluate.py --model_root /path/to/models --output_dir results/
```

For a fully reproducible public example, use:

```text
notebooks/demo_analysis.ipynb
```

---

## Related Projects

* **SafeTraj-Prototype** — Interactive trajectory analysis toolkit with Streamlit dashboard and REST API

---

## Citation

```bibtex
@mastersthesis{bathaei2026safetraj,
  author  = {Bathaei Pourmand, Pouya},
  title   = {Analysis of Neural Trajectory Prediction for Collision Avoidance in Smart Wheelchairs},
  school  = {University of Genoa},
  year    = {2026},
  note    = {MSc in Computer Engineering (AI), REXASI-PRO Project}
}
```

---

## License

MIT License.

See `LICENSE` for details.

The DNN-LNA model weights remain proprietary assets of the REXASI-PRO project and are not covered by this license.
