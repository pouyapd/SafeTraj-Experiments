"""
Presentation figures derived from the published result tables (read-only).

    python tools/derived_figures.py        # -> docs/figures/derived/*.png

Every figure reads only files that tools/verify_results.py shows to be
consistent with the thesis tables, and says in its footnote which file it
uses. Two of them are views the thesis does not show; the 95 % intervals are
an additional calculation (Wilson score interval for 200 inputs per cell)
and are labelled as such. Nothing in results/ is modified.
"""

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.thesis_values import GOALS, MODELS, N_INPUTS, SUCCESS_RATE  # noqa: E402

TABLES = ROOT / "results" / "tables"
OUT = ROOT / "docs" / "figures" / "derived"
SURFACE, INK, INK_2, MUTED, GRID, BASELINE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
MODEL_COLORS = dict(zip(MODELS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
GOAL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
GOAL_LABELS = ["A (0.5, 0.5) near lateral", "B (1.0, 0.0) straight ahead", "C (1.5, −0.5) far off-axis"]
SHORT = {m: m.replace("DNN_LNA_", "") for m in MODELS}


def style():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"], "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "axes.labelcolor": INK_2, "axes.edgecolor": BASELINE, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2, "legend.frameon": False, "legend.fontsize": 9,
        "lines.linewidth": 2.0, "savefig.dpi": 160, "savefig.bbox": "tight",
    })


def note(fig, text, y=-0.02):
    fig.text(0.01, y, text, fontsize=8, color=MUTED, ha="left", va="top")


def wilson(k: int, n: int, conf: float = 0.95):
    z = stats.norm.ppf(0.5 + conf / 2)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def failures(fd, model, goal):
    row = fd[(fd.Model == model) & np.isclose(fd.GoalX, goal[0]) & np.isclose(fd.GoalY, goal[1])]
    return int(row.NumFailures.iloc[0]) if len(row) else 0


def fig_success_intervals(fd):
    order = sorted(MODELS, key=lambda m: SUCCESS_RATE[m][3])
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    offsets = [-0.22, 0.0, 0.22]
    for gi, (goal, color, label) in enumerate(zip(GOALS, GOAL_COLORS, GOAL_LABELS)):
        for mi, m in enumerate(order):
            k = N_INPUTS - failures(fd, m, goal)
            lo, hi = wilson(k, N_INPUTS)
            y = mi + offsets[gi]
            ax.plot([100 * lo, 100 * hi], [y, y], color=color, linewidth=1.6, alpha=0.8)
            ax.plot(100 * k / N_INPUTS, y, "o", color=color, markersize=7, markeredgecolor=SURFACE,
                    label=label if mi == 0 else None)
    for mi, m in enumerate(order):
        ax.text(101.5, mi, f"overall {SUCCESS_RATE[m][3]:.1f} %", va="center", fontsize=8.5, color=INK_2)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([SHORT[m] for m in order])
    ax.set_xlim(0, 100)
    ax.set_xlabel("strict success rate (%)  —  final distance < 0.30 m")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.13), ncol=3, title="goal", title_fontsize=9)
    ax.set_title("Strict success per model and goal, with 95 % intervals")
    note(fig, "Points: thesis Table 4.1 (= 200 − failures in results/tables/exp1_failure_distance_summary.csv). "
              "Additional calculation, not in the thesis: 95 % Wilson intervals for 200 inputs per cell.", y=-0.13)
    fig.savefig(OUT / "success_with_intervals.png")
    plt.close(fig)


def fig_orientation(theta):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), sharey=True)
    for ax, goal, label in zip(axes, GOALS, GOAL_LABELS):
        for m in MODELS:
            d = theta[(theta.Model == m) & np.isclose(theta.GoalX, goal[0]) & np.isclose(theta.GoalY, goal[1])
                      & (theta.Count > 0)].sort_values("Theta_center")
            ax.plot(d.Theta_center, d.SuccessRate, marker="o", markersize=5, color=MODEL_COLORS[m], label=SHORT[m])
        ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        ax.set_xticklabels(["−π", "−π/2", "0", "π/2", "π"])
        ax.set_xlabel("initial orientation θ (rad), bin centre")
        ax.set_title(f"Goal {label}", fontsize=10)
        ax.set_ylim(0, 104)
    axes[0].set_ylabel("strict success rate in bin (%)")
    axes[-1].legend(loc="center left", fontsize=9, bbox_to_anchor=(1.02, 0.5), title="model", title_fontsize=9)
    counts = theta[(theta.Model == MODELS[0]) & np.isclose(theta.GoalX, GOALS[1][0]) & np.isclose(theta.GoalY, GOALS[1][1])
                   & (theta.Count > 0)].sort_values("Theta_center").Count.tolist()
    fig.suptitle("Strict success by initial orientation", x=0.01, ha="left", fontsize=12, fontweight="semibold")
    note(fig, "Drawn from results/tables/exp1_theta_sensitivity.csv: 8 orientation bins with "
              f"{', '.join(map(str, counts))} of the 200 inputs (same inputs for every goal and model); bin-weighted rates "
              "reproduce thesis Table 4.1 (tools/verify_results.py). This view is not a figure of the thesis.")
    fig.savefig(OUT / "orientation_sensitivity.png")
    plt.close(fig)


def fig_failure_severity(fd):
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    order = sorted(MODELS, key=lambda m: SUCCESS_RATE[m][3])
    offsets = [-0.22, 0.0, 0.22]
    for gi, (goal, color, label) in enumerate(zip(GOALS, GOAL_COLORS, GOAL_LABELS)):
        for mi, m in enumerate(order):
            r = fd[(fd.Model == m) & np.isclose(fd.GoalX, goal[0]) & np.isclose(fd.GoalY, goal[1])]
            y = mi + offsets[gi]
            if r.empty:
                ax.text(0.31, y, f"goal {'ABC'[gi]}: no failure", fontsize=7, color=MUTED, va="center")
                continue
            r = r.iloc[0]
            ax.plot([r.MinFailureDist, r.MaxFailureDist], [y, y], color=color, linewidth=1.6, alpha=0.8)
            ax.plot(r.MeanFailureDist, y, "o", color=color, markersize=7, markeredgecolor=SURFACE,
                    label=label if mi == 0 else None)
            ax.text(r.MaxFailureDist + 0.01, y, f"n={int(r.NumFailures)}", fontsize=7, color=MUTED, va="center")
    ax.axvline(0.30, color=BASELINE, linewidth=1.0)
    ax.axvline(0.50, color=BASELINE, linewidth=1.0, linestyle=(0, (3, 2)))
    ax.set_ylim(-0.95, len(order) - 0.5)
    ax.text(0.303, -0.8, "0.30 m: strict threshold", fontsize=7.5, color=INK_2)
    ax.text(0.503, -0.8, "0.50 m: near-success limit", fontsize=7.5, color=INK_2)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([SHORT[m] for m in order])
    ax.set_xlabel("final distance to goal of failed trajectories (m)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper right", title="goal", title_fontsize=9, fontsize=8)
    ax.set_title("How far failures end from the goal (mean, min–max)")
    note(fig, "Drawn from results/tables/exp1_failure_distance_summary.csv (strict failures, d ≥ 0.30 m). "
              "This view is not a figure of the thesis.")
    fig.savefig(OUT / "failure_severity.png")
    plt.close(fig)


def main() -> int:
    style()
    OUT.mkdir(parents=True, exist_ok=True)
    fd = pd.read_csv(TABLES / "exp1_failure_distance_summary.csv")
    theta = pd.read_csv(TABLES / "exp1_theta_sensitivity.csv")
    fig_success_intervals(fd)
    fig_orientation(theta)
    fig_failure_severity(fd)
    for p in sorted(OUT.glob("*.png")):
        print("wrote", p.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
