"""
model_utils.py — Model loading and inference utilities for DNN-LNA models.

The DNN-LNA models are pretrained TensorFlow SavedModels developed by project
partners within the REXASI-PRO framework and are NOT included in this repo.
This module provides the interface to load and run them.
"""

import numpy as np
import tensorflow as tf
from pathlib import Path


def load_models(model_root: str, model_names: list) -> dict:
    """
    Load all DNN-LNA SavedModel signatures from disk.

    Args:
        model_root: Path to the folder containing one subfolder per model.
        model_names: List of model name strings.

    Returns:
        Dict mapping model name -> TF serving signature callable.
    """
    loaded = {}
    for name in model_names:
        path = Path(model_root) / name / "DNN_LNA_model"
        if not path.exists():
            raise FileNotFoundError(
                f"SavedModel not found at: {path}\n"
                "Make sure MODEL_ROOT points to your local model directory."
            )
        model = tf.saved_model.load(str(path))
        sig = model.signatures.get("serving_default")
        if sig is None:
            sig = list(model.signatures.values())[0]
        loaded[name] = sig
        print(f"  Loaded: {name}")

    print(f"\n✅  {len(loaded)} models ready.")
    return loaded


def predict(model_sig, theta_v_omega: np.ndarray,
            goal_xy: tuple, map_shape: tuple) -> np.ndarray:
    """
    Run inference for a batch of inputs against a single goal.

    Args:
        model_sig:      TF serving signature for one model.
        theta_v_omega:  Array of shape (N, 3) — [theta, v, omega] per sample.
        goal_xy:        (gx, gy) target position.
        map_shape:      (H, W, C) shape of the dummy occupancy map.

    Returns:
        Predicted trajectory tensor of shape (N, 5, 30).
        Channels: [x, y, theta, v, omega] at each of 30 time steps.
    """
    N = len(theta_v_omega)
    gx, gy = goal_xy
    H, W, C = map_shape

    input_2 = theta_v_omega.astype(np.float32)                          # (N, 3)
    input_3 = np.tile([[gx, gy, 0.0, 0.0, 0.0]], (N, 1)).astype(np.float32)  # (N, 5)
    input_1 = np.zeros((N, H, W, C), dtype=np.float32)                  # (N, H, W, C)

    out = model_sig(
        input_2=tf.convert_to_tensor(input_2),
        input_3=tf.convert_to_tensor(input_3),
        input_1=tf.convert_to_tensor(input_1),
    )
    y = list(out.values())[0].numpy()

    # Normalise to (N, 5, 30) — some model variants return (N, 30, 5)
    if y.ndim != 3:
        raise ValueError(f"Unexpected output shape: {y.shape}")
    if y.shape[1] == 30 and y.shape[2] == 5:
        y = np.transpose(y, (0, 2, 1))

    return y  # (N, 5, 30)


def predict_batched(model_sig, theta_v_omega: np.ndarray,
                    goal_xy: tuple, map_shape: tuple,
                    batch_size: int = 50) -> np.ndarray:
    """
    Wrapper around predict() that handles large N in batches.
    Returns concatenated output of shape (N, 5, 30).
    """
    parts = []
    N = len(theta_v_omega)
    for start in range(0, N, batch_size):
        batch = theta_v_omega[start: start + batch_size]
        parts.append(predict(model_sig, batch, goal_xy, map_shape))
    return np.concatenate(parts, axis=0)


def final_xy(outputs: np.ndarray) -> np.ndarray:
    """
    Extract the (x, y) position at the last predicted time step.

    Args:
        outputs: Array of shape (N, 5, 30).

    Returns:
        Array of shape (N, 2).
    """
    return outputs[:, :2, -1]


def goal_distances(outputs: np.ndarray, goal_xy: tuple) -> np.ndarray:
    """
    Compute Euclidean distance from the final predicted position to goal_xy.

    Args:
        outputs:  Array of shape (N, 5, 30).
        goal_xy:  (gx, gy) target position.

    Returns:
        Array of shape (N,) — distance per sample.
    """
    gx, gy = goal_xy
    xy = final_xy(outputs)                          # (N, 2)
    return np.linalg.norm(xy - np.array([gx, gy]), axis=1)
