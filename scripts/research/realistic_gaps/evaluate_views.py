"""Score a saved checkpoint and the organisers' baseline on clean / old-corrupted / realistic views.

Only external DFL matches are read. The realistic view follows the BAR-ATH degradation profile
produced by degradation_profile.py (aggregate statistics only, no BAR-ATH frames or labels).

  python scripts/research/realistic_gaps/evaluate_views.py --checkpoint selected.pt \
      --profile data/research/bar_ath_degradation_profile.json --match J03WPY --seeds 3
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

from degradation_profile import profile
from evaluation import possession_metrics as pm
from possession.external import read_sportec
from possession.temporal import symmetric_probability, viterbi_labels
from scripts.prepare_external_v08 import corrupt_tracking
from scripts.run_local import model_from_state
from simulate import degrade, load_profile
from starter.baseline import smooth


def nearest_codes(players, teams, ball):
    d = np.linalg.norm(players - ball[:, None, :], axis=-1)
    d = np.where(np.isfinite(d), d, np.inf)
    best = d.argmin(1)
    codes = teams[best].astype(np.int8)
    codes[~np.isfinite(d.min(1))] = -1
    return codes


def score(truths, preds, balls):
    t, p = [], []
    for a, b in zip(truths, preds):
        t += [a, np.array(["X"])]
        p += [b, np.array(["X"])]
    r = pm.evaluate(pm.label_codes(np.concatenate(t)), pm.label_codes(np.concatenate(p)))
    truth, pred = np.concatenate(truths), np.concatenate(preds)
    seen = np.concatenate([np.isfinite(b).all(1) for b in balls])
    dead = truth == "D"
    return dict(spell_f1=r["spell_f1_mean"], macro_f1=r["macro_f1"], alive_dead_f1=r["alive_dead_f1"],
                gt_spells=r["n_gt_spells"], pred_spells=r["n_pred_spells"],
                ball_missing=float(1 - seen.mean()),
                dead_recall=float((pred[dead] == "D").mean()),
                dead_recall_ball_visible=float((pred[dead & seen] == "D").mean()) if (dead & seen).any() else None,
                dead_recall_ball_missing=float((pred[dead & ~seen] == "D").mean()) if (dead & ~seen).any() else None,
                share_of_missing_ball_frames_dead=float(dead[~seen].mean()) if (~seen).any() else None,
                pred_dead_share=float((pred == "D").mean()), true_dead_share=float(dead.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--sportec", default=str(ROOT / "data/external/sportec"))
    ap.add_argument("--match", default="J03WPY")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--views", nargs="*", help="subset of view names; default all")
    ap.add_argument("--out", default=str(ROOT / "data/research/views.json"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = model_from_state(state, device)
    version = state["config"].get("feature_version")
    builder = __import__("possession.features_" + version, fromlist=["build_features"]).build_features
    prof = load_profile(json.loads(Path(args.profile).read_text()))

    folder = Path(args.sportec)
    halves = read_sportec(next(folder.glob("*matchinformation*" + args.match + ".xml")),
                          next(folder.glob("*positions*" + args.match + ".xml")))
    views = {"clean": lambda h, i: (h.players, h.ball),
             "old_random_corruption": lambda h, i: corrupt_tracking(h.players, h.ball, 1000 + i)}
    for s in range(args.seeds):
        views[f"realistic_seed{s}"] = lambda h, i, s=s: degrade(h.players, h.teams, h.ball, h.pitch, prof, 100 * s + i)
        for mode in ("hide", "carry", "mixed"):
            views[f"unfreeze_{mode}_seed{s}"] = lambda h, i, s=s, mode=mode: degrade(
                h.players, h.teams, h.ball, h.pitch, prof, 100 * s + i, unfreeze_mode=mode)
    if args.views:
        views = {k: v for k, v in views.items() if k in args.views}

    results = {"match": args.match, "decoder_costs": list(state["decoder_costs"]),
               "dfl_clean_profile": [{k: v for k, v in profile(h.players, h.teams, h.ball, h.pitch).items()
                                      if k not in ("loss_hazard_joint_bins", "ball_gap_frames", "player_fragment_frames")}
                                     for h in halves]}
    for name, make in views.items():
        truths, model_preds, base_preds, balls = [], [], [], []
        for i, h in enumerate(halves):
            players, ball = make(h, i)
            x, names = builder(players, h.teams, ball, h.pitch, step=1)
            if names != state["names"]:
                raise ValueError("Feature schema differs from checkpoint")
            prob = symmetric_probability(model, x, names, state["mean"], state["scale"], device)
            model_preds.append(viterbi_labels(prob, *state["decoder_costs"])[:len(h.labels)])
            base_preds.append(smooth(nearest_codes(players, h.teams, ball), 50))
            truths.append(h.labels)
            balls.append(ball)
        results[name] = {"model": score(truths, model_preds, balls), "baseline_smooth50": score(truths, base_preds, balls)}
        print(name, json.dumps(results[name]), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
