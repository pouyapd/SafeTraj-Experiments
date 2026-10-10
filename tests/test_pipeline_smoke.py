"""
Smoke test of the published evaluation pipeline (src/) without TensorFlow and
without the proprietary models.

A minimal stand-in for the `tensorflow` module and a synthetic "model"
signature (a straight line from the start toward the goal, with an error that
grows with |theta|) let the unchanged code in src/model_utils.py and
src/evaluate.py run end to end. Outputs go to a temporary directory only.
This checks the code paths, shapes and labelling logic — not the thesis
results, which require the DNN-LNA models.
"""

import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parent.parent / "src"


class _Out:
    def __init__(self, arr):
        self._arr = arr

    def numpy(self):
        return self._arr


def _fake_signature(layout="N5T"):
    """Synthetic predictor with the DNN-LNA interface: inputs input_1 (map), input_2 (cmd), input_3 (goal)."""
    def sig(input_2, input_3, input_1):
        cmd, goal = np.asarray(input_2), np.asarray(input_3)
        n = len(cmd)
        assert input_1.shape == (n, 300, 300, 1) and goal.shape == (n, 5)
        t = np.linspace(0, 1, 30)
        miss = 0.6 * np.abs(cmd[:, 0]) / np.pi                     # error grows with |theta|
        gx, gy = goal[:, 0] + miss, goal[:, 1]
        traj = np.zeros((n, 5, 30), dtype=np.float32)
        traj[:, 0, :] = gx[:, None] * t
        traj[:, 1, :] = gy[:, None] * t
        if layout == "NT5":
            traj = np.transpose(traj, (0, 2, 1))
        return {"output_0": _Out(traj)}
    return sig


@pytest.fixture
def pipeline(monkeypatch):
    monkeypatch.setattr(sys, "dont_write_bytecode", True)   # leave no __pycache__ in src/
    fake_tf = types.ModuleType("tensorflow")
    fake_tf.convert_to_tensor = lambda x: x
    fake_tf.saved_model = types.SimpleNamespace(load=lambda path: None)
    monkeypatch.setitem(sys.modules, "tensorflow", fake_tf)
    monkeypatch.syspath_prepend(str(SRC))
    for name in ("config", "model_utils", "evaluate"):
        sys.modules.pop(name, None)
    mu = importlib.import_module("model_utils")
    ev = importlib.import_module("evaluate")
    yield mu, ev
    for name in ("config", "model_utils", "evaluate"):
        sys.modules.pop(name, None)


@pytest.mark.parametrize("layout", ["N5T", "NT5"])
def test_predict_normalises_output_layout(pipeline, layout):
    mu, _ = pipeline
    cmd = np.array([[0.0, 1.0, 0.0], [np.pi, 1.0, 0.0]])
    out = mu.predict(_fake_signature(layout), cmd, (1.0, 0.0), (300, 300, 1))
    assert out.shape == (2, 5, 30)
    d = mu.goal_distances(out, (1.0, 0.0))
    np.testing.assert_allclose(d, [0.0, 0.6], atol=1e-6)


def test_batched_prediction_matches_single_call(pipeline):
    mu, _ = pipeline
    cmd = np.random.default_rng(0).uniform(-1, 1, size=(120, 3))
    a = mu.predict_batched(_fake_signature(), cmd, (0.5, 0.5), (300, 300, 1), batch_size=50)
    b = mu.predict(_fake_signature(), cmd, (0.5, 0.5), (300, 300, 1))
    np.testing.assert_allclose(a, b)


def test_experiments_run_end_to_end(pipeline, tmp_path):
    _, ev = pipeline
    models = {"synthetic_model": _fake_signature()}
    df1 = ev.run_exp1(models, tmp_path)
    inputs = np.load(tmp_path / "exp1_inputs.npy")
    assert inputs.shape == (ev.config.NUM_SAMPLES_EXP1, 3)
    lo = [r[0] for r in ev.config.INPUT_RANGES.values()]
    hi = [r[1] for r in ev.config.INPUT_RANGES.values()]
    assert np.all(inputs >= lo) and np.all(inputs <= hi)
    assert len(df1) == ev.config.NUM_SAMPLES_EXP1 * len(ev.config.GOALS)
    assert (df1["Success"] == (df1["Dist"] < ev.config.STRICT_THRESH).astype(int)).all()
    # the same inputs are reused for every goal
    first = df1.groupby(["GoalX", "GoalY"])["Theta"].apply(list)
    assert all(v == first.iloc[0] for v in first)

    df2 = ev.run_exp2(models, tmp_path)
    assert len(df2) == len(ev.config.GOALS)
    shares = df2["SuccessRate(%)"] + df2["NearSuccess(%)"] + df2["FailureRate(%)"]
    np.testing.assert_allclose(shares, 100.0, atol=0.02)
    # with this synthetic model, success means |theta| < 0.30 / 0.6 * pi
    expected = (np.abs(inputs[:, 0]) < 0.5 * np.pi).mean() * 100
    np.testing.assert_allclose(df2["SuccessRate(%)"], round(expected, 2), atol=0.01)

    ev.plot_exp1_input_sensitivity(df1, tmp_path / "fig1")
    ev.plot_exp2_goal_difficulty(df2, tmp_path / "fig2")
    assert len(list((tmp_path / "fig1").glob("*.png"))) == 3
    assert (tmp_path / "fig2" / "exp2_goal_difficulty_map.png").is_file()
