"""Estimate the effect of broadcast ball gap-filling on the J03WPY proxy.

Inside each degraded dev view, a ball-missing frame gets the true DFL ball plus BAR-ATH-sized error
when a simulated broadcast view is live and a Bernoulli(fill_rate_given_live) draw succeeds. Views,
fill rate and error quantiles come from view_model.json (label-free BAR-ATH statistics).

  python scripts/research/broadcast/proxy_ball_fill.py --checkpoint ft/selected.pt --view-model view_model.json \
      --profile data/research/bar_ath_degradation_profile.json
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT / "scripts/research/realistic_gaps")]

import numpy as np
import torch

from evaluate_views import score
from finetune import SPEC, halves_of, view_tracking
from possession.temporal import symmetric_probability, viterbi_labels
from proxy_view import simulate_views
from simulate import load_profile


def build(args, view, i, seed):
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    builder = __import__("possession.features_" + state["config"]["feature_version"], fromlist=["x"]).build_features
    vm = json.loads(Path(args.view_model).read_text())
    h = halves_of("J03WPY", args.sportec)[i]
    players, ball = view_tracking(h, view, load_profile(json.loads(Path(args.profile).read_text())), SPEC[view][2] + i, args)
    rng = np.random.default_rng(seed)
    n = len(h.labels)
    views = simulate_views(h.labels == "D", vm["transitions"], seed)
    missing = ~np.isfinite(ball[:n]).all(1)
    fill = missing & (views == 0) & (rng.random(n) < vm["fill_rate_given_live"]) & np.isfinite(h.ball[:n]).all(1)
    radius = np.interp(rng.random(n), np.linspace(0, 1, 101), vm["error_quantiles"])
    angle = rng.random(n) * 2 * np.pi
    ball = ball.copy()
    ball[:n][fill] = h.ball[:n][fill] + np.c_[radius * np.cos(angle), radius * np.sin(angle)][fill]
    x, _ = builder(players, h.teams, ball, h.pitch, step=1)
    return view, i, seed, x[:n].astype(np.float32), ball[:n], h.labels, int(fill.sum()), int(missing.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--view-model", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--sportec", default=str(ROOT / "data/external/sportec"))
    ap.add_argument("--cache", default="/content/ft-cache")
    ap.add_argument("--views", nargs="+", default=["stateA", "stateB"])
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--stoppage-q", type=float, default=0.85)
    ap.add_argument("--stoppage-fraction", type=float, default=1.0)
    ap.add_argument("--max-alive-gap", type=float, default=20.0)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    from scripts.run_local import model_from_state
    model = model_from_state(state, device)
    costs = state["decoder_costs"]
    suffix = f"_q{args.stoppage_q:g}_f{args.stoppage_fraction:g}_cap{args.max_alive_gap:g}"

    def predict(x):
        return viterbi_labels(symmetric_probability(model, x, state["names"], state["mean"], state["scale"], device), *costs)

    base = {}
    for view in args.views:
        halves = [(np.load(Path(args.cache) / f"J03WPY_{i}_{view}{suffix}.npy"),
                   np.load(Path(args.cache) / f"J03WPY_{i}_{view}{suffix}_ball.npy")) for i in range(2)]
        truth = [np.array(["H", "A", "D", "X"])[np.load(Path(args.cache) / f"J03WPY_{i}_y.npy")] for i in range(2)]
        base[view] = score(truth, [predict(x)[:len(t)] for (x, _), t in zip(halves, truth)], [b for _, b in halves])["spell_f1"]
        print("base", view, round(base[view], 4), flush=True)
    jobs = [(v, i, 1000 * s + i) for v in args.views for s in range(args.seeds) for i in range(2)]
    built = {}
    with ProcessPoolExecutor(args.workers) as pool:
        for view, i, seed, x, ball, labels, filled, missing in pool.map(build, *zip(*[(args, *j) for j in jobs])):
            built[(view, seed - i, i)] = (x, ball, labels)
            print("built", view, i, seed, "filled", filled, "of missing", missing, flush=True)
    gains = []
    for view in args.views:
        for s in range(args.seeds):
            parts = [built[(view, 1000 * s, i)] for i in range(2)]
            r = score([p[2] for p in parts], [predict(p[0])[:len(p[2])] for p in parts], [p[1] for p in parts])
            gains.append(r["spell_f1"] - base[view])
            print("fill", view, s, round(r["spell_f1"], 4), "gain", round(gains[-1], 4), flush=True)
    print("MEAN_GAIN", round(float(np.mean(gains)), 4), "MIN", round(float(np.min(gains)), 4))


if __name__ == "__main__":
    main()
