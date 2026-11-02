"""Run the selected model's competition inference with tactical ball gaps filled from the aligned broadcast.

A tactical frame with no ball takes the broadcast ball when the broadcast frame is a live wide view
(homography valid, >= 6 players, player-match cost < 1.5 m). No labels are read.

  python scripts/research/broadcast/apply_ball_fill.py --config configs/local_v26_pretrained_attention.json \
      --out data/local-v11/ball-fill
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE), str(ROOT)]

import numpy as np
import torch

import possession.features_v25 as features_v25
import possession.inference_cache as inference_cache
import scripts.run_local as run_local
from games import GAME, half_video_blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--signal", default="/tmp/bcast_signal.npz")
    args = ap.parse_args()
    signal = np.load(args.signal)
    live = signal["covered"] & signal["valid"] & (signal["players"] >= 6) & (signal["cost"] < 1.5)
    broadcast_ball = np.where((live & np.isfinite(signal["ball"]).all(1))[:, None], signal["ball"], np.nan).astype(np.float32)
    blocks = iter(half_video_blocks(GAME))
    original = features_v25.build_features
    filled = []

    def build(players, teams, ball, pitch, step):
        _, first, last = next(blocks)
        extra = broadcast_ball[first:last + 1]
        gap = ~np.isfinite(ball).all(1) & np.isfinite(extra).all(1)
        ball = ball.copy()
        ball[gap] = extra[gap]
        filled.append(int(gap.sum()))
        return original(players, teams, ball, pitch, step=step)

    features_v25.build_features = build
    inference_cache.cached_block = lambda root, key, generate, rows: (*generate(), None)
    out = Path(args.out)
    config = json.loads(Path(args.config).read_text())
    run_local.infer_competition(config, out, torch.device("cpu"))
    (out / "ball_fill.json").write_text(json.dumps(dict(filled_frames_per_block=filled), indent=1))
    print("filled", filled, sum(filled))


if __name__ == "__main__":
    main()
