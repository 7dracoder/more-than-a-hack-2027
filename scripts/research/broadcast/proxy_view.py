"""Estimate the effect of the broadcast-view ratio on the J03WPY proxy.

Views are simulated per DFL state with the per-state Markov chains estimated label-free on BAR-ATH
(view_model.json), then P(D) is multiplied by view_ratio ** weight before decoding.

  python scripts/research/broadcast/proxy_view.py --checkpoint ft/selected.pt --view-model view_model.json \
      --cache /content/ft-cache --views stateA_q0.85_f1_cap20 stateB_q0.85_f1_cap20
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT / "scripts/research/realistic_gaps")]

import numpy as np
import torch

from evaluate_views import score
from possession.temporal import symmetric_probability, viterbi_labels
from scripts.run_local import model_from_state


def simulate_views(dead, transitions, seed):
    rng = np.random.default_rng(seed)
    chains = {k: np.cumsum(np.asarray(v), 1) for k, v in transitions.items()}
    view = np.zeros(len(dead), int)
    u = rng.random(len(dead))
    for t in range(1, len(dead)):
        row = chains["stop" if dead[t] else "play"][view[t - 1]]
        view[t] = min(int(np.searchsorted(row, u[t])), 2)
    return view


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--view-model", required=True)
    ap.add_argument("--cache", default="/content/ft-cache")
    ap.add_argument("--views", nargs="+", required=True)
    ap.add_argument("--weights", nargs="+", type=float, default=[0, 0.5, 1, 1.5, 2])
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = model_from_state(state, device)
    vm = json.loads(Path(args.view_model).read_text())
    ratio = np.asarray(vm["view_ratio"])
    cache = Path(args.cache)
    truth = [np.array(["H", "A", "D", "X"])[np.load(cache / f"J03WPY_{i}_y.npy")] for i in range(2)]
    results = {}
    for view_name in args.views:
        halves = [dict(x=np.load(cache / f"J03WPY_{i}_{view_name}.npy"), ball=np.load(cache / f"J03WPY_{i}_{view_name}_ball.npy"))
                  for i in range(2)]
        probs = [symmetric_probability(model, h["x"], state["names"], state["mean"], state["scale"], device)[:len(truth[i])]
                 for i, h in enumerate(halves)]
        for seed in range(args.seeds):
            views = [simulate_views(truth[i] == "D", vm["transitions"], 1000 * seed + i) for i in range(2)]
            for w in args.weights:
                preds = []
                for p, v in zip(probs, views):
                    q = p.copy()
                    q[:, 2] *= ratio[v] ** w
                    q[:, :3] /= q[:, :3].sum(1, keepdims=True)
                    preds.append(viterbi_labels(q, *state["decoder_costs"]))
                r = score(truth, preds, [h["ball"] for h in halves])
                results.setdefault(w, []).append(r["spell_f1"])
                print(view_name, seed, w, round(r["spell_f1"], 4), round(r["macro_f1"], 4), round(r["pred_dead_share"], 3), flush=True)
    print("MEAN", {w: round(float(np.mean(v)), 4) for w, v in results.items()})


if __name__ == "__main__":
    main()
