"""Run the selected model's competition inference with a broadcast-view likelihood ratio on D.

The broadcast timeline is aligned to tactical by fixed per-half offsets found from player positions
(fine.py). Each graded tactical frame gets a view class (live / close-up / other); an unsupervised
latent-class model (latent.py) gives P(view | stoppage) / P(view | play). Before Viterbi decoding,
P(D) is multiplied by that ratio and the H/A/D probabilities are renormalized. No labels are read.

  python scripts/research/broadcast/apply_view_prior.py --config configs/local_v26_pretrained_attention.json \
      --out data/local-v11/view-prior --data /tmp/mtah_data
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

import scripts.run_local as run_local
from games import GAME, half_video_blocks
from latent import fit, indicators
from load import load


def view_ratios(data_dir, signal_path):
    signal = dict(np.load(signal_path))
    tp, tb, _, _, _ = load("tactical", data_dir)
    graded = np.zeros(len(tb), bool)
    for _, a, b in half_video_blocks(GAME):
        graded[a:b + 1] = True
    ball, view, speed = indicators(signal, tp, tb)
    keep = graded & signal["covered"]
    prior, cond, _ = fit([ball[keep], view[keep], speed[keep]], [2, 3, 3])
    ratio = cond[1][1] / cond[1][0]
    frame_ratio = np.where(signal["covered"], ratio[view], 1.0)
    return frame_ratio, dict(stoppage_prior=float(prior), view_ratio=ratio.tolist(),
                             view_given_play=cond[1][0].tolist(), view_given_stoppage=cond[1][1].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="/tmp/mtah_data", help="folder with tactical/ and broadcast/")
    ap.add_argument("--signal", default="/tmp/bcast_signal.npz")
    args = ap.parse_args()
    frame_ratio, audit = view_ratios(args.data, args.signal)
    blocks = iter(half_video_blocks(GAME))
    original = run_local.viterbi_labels

    def decode(probability, *costs):
        _, first, last = next(blocks)
        p = probability.copy()
        p[:, 2] *= frame_ratio[first:last + 1]
        p[:, :3] /= p[:, :3].sum(1, keepdims=True)
        return original(p, *costs)

    run_local.viterbi_labels = decode
    out = Path(args.out)
    config = json.loads(Path(args.config).read_text())
    run_local.infer_competition(config, out, torch.device("cpu"))
    (out / "view_prior.json").write_text(json.dumps(audit, indent=1))
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
