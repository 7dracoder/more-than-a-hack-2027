"""Exploratory camera-framing signal probe; not a possession predictor.

Fit observed ball coordinates from inverse-projected image-center coordinates in
the first graded half, then evaluate observed coordinates in the second half.
Only noisy provided ATD observations are used; no possession truth is read.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from games import GAMES, GAME, half_video_blocks, require  # noqa: E402
from starter.load_homography import load  # noqa: E402


def run():
    matrices, _, valid = load()
    n = len(matrices)
    center = np.full((n, 2), np.nan)
    image_center = np.array([960.0, 540.0, 1.0])
    for start in range(0, n, 10000):
        stop = min(n, start + 10000)
        indices = np.flatnonzero(valid[start:stop]) + start
        points = np.linalg.inv(matrices[indices]) @ image_center
        center[indices] = points[:, :2] / points[:, 2, None]

    ball = np.full((n, 2), np.nan)
    raw_path = require(["atd_raw"])[0]
    with open(raw_path) as handle:
        for line in handle:
            parts = line.strip().split(":")
            if len(parts) != 3:
                continue
            values = parts[2].split(",")
            if len(values) < 2:
                continue
            try:
                ball[int(parts[0]) - 1] = [float(values[0]), float(values[1])]
            except ValueError:
                continue

    finite = np.isfinite(center).all(1) & np.isfinite(ball).all(1) & valid
    finite &= (np.abs(center) < 5).all(1)
    finite &= (ball >= -0.2).all(1) & (ball <= 1.2).all(1)
    masks = []
    blocks = half_video_blocks(GAME)
    for _, start, stop in blocks:
        mask = finite.copy()
        mask[:start] = False
        mask[stop + 1:] = False
        masks.append(mask)
    train, test = masks
    if train.sum() < 3 or test.sum() < 3:
        raise ValueError("Insufficient finite observed-ball samples in the halves")

    features = np.column_stack([np.ones(n), center])
    coefficients = np.linalg.lstsq(features[train], ball[train], rcond=None)[0]
    scale = np.asarray(GAMES[GAME]["pitch"])
    predicted = features[test] @ coefficients
    truth = ball[test]
    errors = np.linalg.norm((predicted - truth) * scale, axis=1)
    null = np.median(ball[train], axis=0)
    null_errors = np.linalg.norm((truth - null) * scale, axis=1)

    # Break temporal synchronization within the test half. This is a diagnostic
    # control only; a fixed shift may retain some low-frequency match structure.
    _, test_start, test_stop = blocks[1]
    shifted = center.copy()
    shifted[test_start:test_stop + 1] = np.roll(
        center[test_start:test_stop + 1], 1500, axis=0)
    control_mask = test & np.isfinite(shifted).all(1)
    shifted_features = np.column_stack([np.ones(n), shifted])
    control_prediction = shifted_features[control_mask] @ coefficients
    control_errors = np.linalg.norm(
        (control_prediction - ball[control_mask]) * scale, axis=1)

    return {
        "experiment": "camera_ground_center_ball_location_probe",
        "purpose": "Exploratory signal check, not possession or natural-gap evaluation",
        "train": "finite bounded ATD ball observations in first graded half",
        "test": "finite bounded ATD ball observations in second graded half",
        "features": ["intercept", "inverse-projected image-center x", "inverse-projected image-center y"],
        "image_dimensions": [1920, 1080],
        "model": "ordinary least squares; no hyperparameter search",
        "filters": {"absolute_center_coordinate_below": 5, "ball_coordinate_bounds": [-0.2, 1.2]},
        "train_count": int(train.sum()),
        "test_count": int(test.sum()),
        "test_mean_error_m": float(errors.mean()),
        "test_median_error_m": float(np.median(errors)),
        "test_axis_mean_absolute_error_m": np.abs((predicted - truth) * scale).mean(0).tolist(),
        "train_median_ball_null_test_mean_error_m": float(null_errors.mean()),
        "camera_ball_x_correlation_test": float(np.corrcoef(center[test, 0], ball[test, 0])[0, 1]),
        "camera_ball_y_correlation_test": float(np.corrcoef(center[test, 1], ball[test, 1])[0, 1]),
        "test_camera_shift_control_seconds": 60,
        "test_camera_shift_control_count": int(control_mask.sum()),
        "test_camera_shift_control_mean_error_m": float(control_errors.mean()),
        "limitations": [
            "ATD observations are noisy, not corrected ball ground truth",
            "only visible-ball frames tested; natural gaps can differ systematically",
            "one-match half split, not cross-match validation",
            "ground-plane image center is a framing proxy, not camera pose or human gaze",
            "no possession labels used or scored",
            "circular-shift control retains some low-frequency structure",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "data/research/camera-probe.json")
    args = parser.parse_args()
    report = run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
