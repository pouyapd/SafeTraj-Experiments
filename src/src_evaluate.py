"""
evaluate.py — Main evaluation pipeline for SafeTraj-Experiments.

Runs Experiment 1 (input-space sensitivity) and Experiment 2 (goal-based
difficulty analysis) for all five DNN-LNA neural trajectory prediction models.

Usage:
    python evaluate.py --model_root /path/to/models --output_dir results/

The neural models are NOT included in this repository (proprietary weights).
See README for instructions on obtaining them via the REXASI-PRO project.
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for server / CI runs

import config
from model_utils import load_models, predict_batched, goal_distances


# ── Reproducibility ──────────────────────────────────────────────────────────
np.random.seed(config.SEED)
random.seed(config.SEED)


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 1 — Input-Space Sensitivity Analysis
# ─────────────────────────────────────────────────────────────────────────────

def run_exp1(models: dict, output_dir: Path) -> pd.DataFrame:
    """
    Sample N random inputs uniformly from the operational range,
    run inference for every model × goal combination, and compute
    per-sample success/failure labels and distances.

    Returns:
        DataFrame with columns:
        [Theta, Vel, Omega, GoalX, GoalY, Model, Dist, Success]
    """
    # ── Sample inputs ──────────────────────────────────────────────────────
    N = config.NUM_SAMPLES_EXP1
    theta = np.random.uniform(*config.INPUT_RANGES["theta"], size=N)
    v     = np.random.uniform(*config.INPUT_RANGES["v"],     size=N)
    omega = np.random.uniform(*config.INPUT_RANGES["omega"], size=N)
    inputs = np.column_stack([theta, v, omega])             # (N, 3)

    np.save(output_dir / "exp1_inputs.npy", inputs)
    print(f"Exp1: {N} inputs sampled and saved.")

    # ── Inference ──────────────────────────────────────────────────────────
    rows = []
    for goal_xy in config.GOALS:
        gx, gy = goal_xy
        print(f"  Goal ({gx}, {gy}) ...")
        for mname, msig in models.items():
            preds = predict_batched(
                msig, inputs, goal_xy,
                config.MAP_SHAPE, config.BATCH_SIZE
            )                                               # (N, 5, 30)
            dists = goal_distances(preds, goal_xy)          # (N,)
            for i in range(N):
                rows.append({
                    "Theta":   float(inputs[i, 0]),
                    "Vel":     float(inputs[i, 1]),
                    "Omega":   float(inputs[i, 2]),
                    "GoalX":   gx,
                    "GoalY":   gy,
                    "Model":   mname,
                    "Dist":    float(dists[i]),
                    "Success": int(dists[i] < config.STRICT_THRESH),
                })

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "exp1_samples.csv", index=False)
    print(f"Exp1: saved {len(df)} rows -> exp1_samples.csv")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 2 — Goal-Based Difficulty Analysis
# ─────────────────────────────────────────────────────────────────────────────

def run_exp2(models: dict, output_dir: Path) -> pd.DataFrame:
    """
    Evaluate all models across the three goal configurations using the
    same N inputs from Exp1. Computes strict and soft success distributions.

    Returns:
        DataFrame with columns:
        [GoalX, GoalY, Model, SuccessRate, NearSuccessRate, FailureRate, AvgDist]
    """
    inputs = np.load(output_dir / "exp1_inputs.npy")   # reuse Exp1 inputs
    N = len(inputs)
    rows = []

    for goal_xy in config.GOALS:
        gx, gy = goal_xy
        print(f"  Goal ({gx}, {gy}) ...")
        for mname, msig in models.items():
            preds = predict_batched(
                msig, inputs, goal_xy,
                config.MAP_SHAPE, config.BATCH_SIZE
            )
            dists = goal_distances(preds, goal_xy)

            strict   = (dists < config.STRICT_THRESH).mean() * 100
            near     = ((dists >= config.STRICT_THRESH) &
                        (dists <  config.NEAR_THRESH)).mean() * 100
            failure  = (dists >= config.NEAR_THRESH).mean() * 100

            rows.append({
                "GoalX":           gx,
                "GoalY":           gy,
                "Model":           mname,
                "SuccessRate(%)":  round(strict,  2),
                "NearSuccess(%)":  round(near,    2),
                "FailureRate(%)":  round(failure, 2),
                "AvgDist(m)":      round(float(dists.mean()), 4),
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "exp2_goal_summary.csv", index=False)
    print(f"Exp2: saved {len(df)} rows -> exp2_goal_summary.csv")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def plot_exp1_input_sensitivity(df: pd.DataFrame, fig_dir: Path):
    """Scatter plots: each input variable vs final distance for all models."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    cols  = {"Theta": "Theta (rad)",
             "Vel":   "Linear velocity (m/s)",
             "Omega": "Angular velocity (rad/s)"}

    for var, xlabel in cols.items():
        plt.figure(figsize=(7, 4))
        ok   = df["Success"] == 1
        fail = ~ok
        plt.scatter(df.loc[ok,   var], df.loc[ok,   "Dist"],
                    s=18, alpha=0.6, color="tab:green", label="Success")
        plt.scatter(df.loc[fail, var], df.loc[fail, "Dist"],
                    s=18, alpha=0.6, color="tab:red",   label="Failure")
        plt.axhline(config.STRICT_THRESH, color="k",
                    linestyle="--", linewidth=1, label=f"d={config.STRICT_THRESH} m")
        plt.xlabel(xlabel)
        plt.ylabel("Final distance to goal (m)")
        plt.title(f"Exp1 — {var} vs distance (all models & goals)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(fig_dir / f"exp1_{var.lower()}_vs_distance.png", dpi=200)
        plt.close()
        print(f"  Saved: exp1_{var.lower()}_vs_distance.png")


def plot_exp2_goal_difficulty(df: pd.DataFrame, fig_dir: Path):
    """Heatmap of average success rate per goal, averaged across all models."""
    fig_dir.mkdir(parents=True, exist_ok=True)

    goal_mean = (
        df.groupby(["GoalX", "GoalY"], as_index=False)["SuccessRate(%)"].mean()
    )

    pivot = goal_mean.pivot(index="GoalY", columns="GoalX",
                            values="SuccessRate(%)")

    plt.figure(figsize=(5, 4))
    im = plt.imshow(pivot.values, origin="lower",
                    cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    plt.colorbar(im, label="Average success rate (%)")
    plt.title("Exp2 — Goal difficulty map (avg over all models)")
    plt.xlabel("Goal X (m)")
    plt.ylabel("Goal Y (m)")
    plt.xticks(range(len(pivot.columns)),
               [f"{x:.1f}" for x in pivot.columns])
    plt.yticks(range(len(pivot.index)),
               [f"{y:.1f}" for y in pivot.index])
    for i, y in enumerate(pivot.index):
        for j, x in enumerate(pivot.columns):
            val = pivot.loc[y, x]
            plt.text(j, i, f"{val:.1f}%", ha="center", va="center",
                     color="black", fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(fig_dir / "exp2_goal_difficulty_map.png", dpi=200)
    plt.close()
    print("  Saved: exp2_goal_difficulty_map.png")


def plot_exp2_success_heatmap(df: pd.DataFrame, fig_dir: Path):
    """Per-model per-goal strict success rate heatmap."""
    try:
        import seaborn as sns
    except ImportError:
        print("seaborn not installed — skipping success heatmap")
        return

    pivot = df.pivot_table(
        index="Model", columns=["GoalX", "GoalY"],
        values="SuccessRate(%)", aggfunc="mean"
    )
    plt.figure(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGnBu",
                linewidths=0.5, linecolor="gray")
    plt.title("Exp2 — Strict success rate (%) per model and goal")
    plt.tight_layout()
    plt.savefig(fig_dir / "exp2_success_heatmap.png", dpi=200)
    plt.close()
    print("  Saved: exp2_success_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run SafeTraj evaluation experiments."
    )
    parser.add_argument(
        "--model_root", required=True,
        help="Path to folder containing one subfolder per DNN-LNA model."
    )
    parser.add_argument(
        "--output_dir", default="results",
        help="Root folder for CSVs and figures (default: results/)."
    )
    args = parser.parse_args()

    out     = Path(args.output_dir)
    tables  = out / "tables"
    fig_e1  = out / "figures_exp1"
    fig_e2  = out / "figures_exp2"
    for d in [tables, fig_e1, fig_e2]:
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SafeTraj — Neural Trajectory Evaluation Pipeline")
    print("=" * 60)
    print(f"Model root : {args.model_root}")
    print(f"Output dir : {out}")
    print()

    # ── Load models ──────────────────────────────────────────────────────
    print("Loading models ...")
    models = load_models(args.model_root, config.MODEL_NAMES)

    # ── Experiment 1 ─────────────────────────────────────────────────────
    print("\n[Experiment 1] Input-space sensitivity analysis")
    exp1_df = run_exp1(models, tables)
    print("\nPlotting Exp1 figures ...")
    plot_exp1_input_sensitivity(exp1_df, fig_e1)

    # ── Experiment 2 ─────────────────────────────────────────────────────
    print("\n[Experiment 2] Goal-based difficulty analysis")
    exp2_df = run_exp2(models, tables)
    print("\nPlotting Exp2 figures ...")
    plot_exp2_goal_difficulty(exp2_df, fig_e2)
    plot_exp2_success_heatmap(exp2_df, fig_e2)

    print("\n✅  All done. Results saved to:", out)


if __name__ == "__main__":
    main()
