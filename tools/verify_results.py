"""
Read-only consistency checks of the published thesis results.

Each check recomputes a reported thesis number from a result file in
results/tables/ (or compares two files) and reports whether they agree. The
script only reads files; it never modifies results/ or src/.

    python tools/verify_results.py            # report; exit code 1 if a check fails

Checks:
  C1  per-goal strict success (Table 4.1) = (200 - failures) / 200 from exp1_failure_distance_summary.csv
  C2  overall strict success (Table 4.1) = mean over the three goals = combined_overview.csv column Exp1
  C3  goal B success and average distance (Tables 4.1, 4.2) = exp1_input_sensitivity_goal_1_0.csv
  C4  overall average distance (Table 4.2) = mean over goals = summary_avg_success.csv (Exp1, AvgDist)
  C5  decision-tree class counts (Success / Failure of 600) = totals of the failure counts
  C6  orientation-bin success rates, weighted by bin counts, = per-goal success (Table 4.1)
  C7  every model-goal cell of exp1_theta_sensitivity.csv has 200 samples, binned identically for all models
  C8  exp1_bad_theta_bins.csv = the bins of exp1_theta_sensitivity.csv with success below 80 %
"""

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.thesis_values import (AVG_DISTANCE, GOALS, MODELS, N_INPUTS, SUCCESS_RATE,  # noqa: E402
                                 TREE_CLASS_COUNTS)

TABLES = ROOT / "results" / "tables"
Check = Tuple[str, bool, str]


def _failures(fd: pd.DataFrame, model: str, goal) -> int:
    row = fd[(fd.Model == model) & np.isclose(fd.GoalX, goal[0]) & np.isclose(fd.GoalY, goal[1])]
    return int(row.NumFailures.iloc[0]) if len(row) else 0   # a missing row means no failure


def run_checks() -> List[Check]:
    fd = pd.read_csv(TABLES / "exp1_failure_distance_summary.csv")
    co = pd.read_csv(TABLES / "combined_overview.csv").set_index("Model")
    sens = pd.read_csv(TABLES / "exp1_input_sensitivity_goal_1_0.csv").set_index("Model")
    avg = pd.read_csv(TABLES / "summary_avg_success.csv")
    theta = pd.read_csv(TABLES / "exp1_theta_sensitivity.csv")
    bad = pd.read_csv(TABLES / "exp1_bad_theta_bins.csv")
    checks: List[Check] = []

    for m in MODELS:
        per_goal = [100.0 * (N_INPUTS - _failures(fd, m, g)) / N_INPUTS for g in GOALS]
        reported = SUCCESS_RATE[m]
        ok = all(round(v, 1) == r for v, r in zip(per_goal, reported[:3]))
        checks.append(("C1", ok, f"{m}: per-goal success {per_goal} vs reported {list(reported[:3])}"))

        overall = float(np.mean(per_goal))
        ok = round(overall, 1) == reported[3] and abs(co.loc[m, "Exp1"] - overall) < 1e-9
        checks.append(("C2", ok, f"{m}: overall {overall:.4f} (combined_overview Exp1 {co.loc[m, 'Exp1']:.4f}) "
                                 f"vs reported {reported[3]}"))

        ok = (round(sens.loc[m, "SuccessRate(%)"], 1) == reported[1]
              and round(sens.loc[m, "AvgDist"], 3) == AVG_DISTANCE[m][1]
              and int(sens.loc[m, "NumFailures"]) == _failures(fd, m, GOALS[1]))
        checks.append(("C3", ok, f"{m}: goal B success {sens.loc[m, 'SuccessRate(%)']}, avg distance "
                                 f"{sens.loc[m, 'AvgDist']} vs reported {reported[1]}, {AVG_DISTANCE[m][1]}"))

        a = avg[(avg.Exp == "Exp1") & (avg.Model == m)].AvgDist.iloc[0]
        mean_goals = float(np.mean(AVG_DISTANCE[m][:3]))
        ok = round(a, 3) == AVG_DISTANCE[m][3] and abs(mean_goals - AVG_DISTANCE[m][3]) < 0.0015
        checks.append(("C4", ok, f"{m}: summary_avg_success AvgDist {a:.4f}, mean of goals {mean_goals:.4f} "
                                 f"vs reported {AVG_DISTANCE[m][3]}"))

        fails = sum(_failures(fd, m, g) for g in GOALS)
        ok = (3 * N_INPUTS - fails, fails) == TREE_CLASS_COUNTS[m]
        checks.append(("C5", ok, f"{m}: success/failure {3 * N_INPUTS - fails}/{fails} vs reported "
                                 f"{TREE_CLASS_COUNTS[m][0]}/{TREE_CLASS_COUNTS[m][1]}"))

        for g, r in zip(GOALS, reported[:3]):
            t = theta[(theta.Model == m) & np.isclose(theta.GoalX, g[0]) & np.isclose(theta.GoalY, g[1])
                      & (theta.Count > 0)]
            weighted = float((t.SuccessRate * t.Count).sum() / t.Count.sum())
            # bin rates are stored with one decimal, so allow 0.05 percentage points of rounding
            checks.append(("C6", abs(weighted - r) <= 0.05,
                           f"{m} goal {g}: bin-weighted success {weighted:.3f} vs reported {r}"))

    populated = theta[theta.Count > 0]
    counts_ok = populated.groupby(["Model", "GoalX", "GoalY"]).Count.sum().eq(N_INPUTS).all()
    bins = populated.groupby(["GoalX", "GoalY", "Theta_center"]).Count.nunique().eq(1).all()
    cells = populated.groupby(["Model", "GoalX", "GoalY"]).ngroups
    checks.append(("C7", bool(counts_ok and bins and cells == len(MODELS) * len(GOALS)),
                   f"{cells} populated model-goal cells, 200 samples each: {bool(counts_ok)}, "
                   f"same bin counts for every model: {bool(bins)}"))

    cols = ["Model", "GoalX", "GoalY", "Theta_center", "SuccessRate", "Count"]
    expected_bad = populated[populated.SuccessRate < 80][cols].reset_index(drop=True)
    same = expected_bad.round(9).equals(bad[cols].round(9).reset_index(drop=True))
    checks.append(("C8", bool(same), f"{len(bad)} rows in exp1_bad_theta_bins.csv; {len(expected_bad)} bins "
                                     f"below 80 % success in exp1_theta_sensitivity.csv"))
    return checks


def main() -> int:
    checks = run_checks()
    for name, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    failed = [c for c in checks if not c[1]]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
