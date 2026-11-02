"""Fine-tune a residual checkpoint's corrector on DFL views without the parked-ball stoppage cue.

Training: six external DFL matches, each half in native plus degraded views (see SPEC; "state" views
tie ball loss to DFL stoppages with --stoppage-q). Selection: mean spell F1 over J03WPY's degraded
dev views. The teacher stays frozen; normalization is the parent's.

  python scripts/research/realistic_gaps/finetune.py --checkpoint selected.pt \
      --profile data/research/bar_ath_degradation_profile.json --out /content/ft-run
"""
import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE), str(ROOT)]

import numpy as np
import torch

from evaluate_views import score
from evaluation import possession_metrics as pm
from possession.external import read_sportec
from possession.temporal import frame_loss, normalized, symmetric_probability, training_weights, viterbi_labels
from simulate import degrade, load_profile

TRAIN = ["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WQQ", "J03WR9"]
# view -> (degradation mode, train seed offset, dev seed offset); "state" ties ball loss to DFL stoppages
SPEC = {"native": (None, 0, 0), "clean": (None, 0, 0), "carry": ("carry", 5, 0), "mixed": ("mixed", 5, 0),
        "stateA": ("state", 7, 0), "stateB": ("state", 8, 100)}
GAP_BUCKETS = [(0, 5), (5, 10), (10, 20), (20, 60), (60, 10_000)]


def halves_of(match, sportec):
    folder = Path(sportec)
    return read_sportec(next(folder.glob("*matchinformation*" + match + ".xml")),
                        next(folder.glob("*positions*" + match + ".xml")))


def file_name(view, args):
    if SPEC[view][0] != "state":
        return view
    suffix = f"_q{args.stoppage_q:g}"
    if args.stoppage_fraction != 1 or args.max_alive_gap:
        suffix += f"_f{args.stoppage_fraction:g}_cap{args.max_alive_gap:g}"
    return view + suffix


def view_tracking(h, view, prof, seed, args):
    mode = SPEC[view][0]
    if mode is None:
        return h.players, h.ball
    if mode == "state":
        return degrade(h.players, h.teams, h.ball, h.pitch, prof, seed, "carry", h.labels, args.stoppage_q,
                       args.stoppage_fraction, round(25 * args.max_alive_gap) if args.max_alive_gap else None)
    return degrade(h.players, h.teams, h.ball, h.pitch, prof, seed, unfreeze_mode=mode)


def dead_by_gap(preds, balls):
    """Predicted dead share inside ball-missing gaps by gap length, and with the ball visible (label-free)."""
    pred, ball = np.concatenate(preds), np.concatenate(balls)
    seen = np.isfinite(ball).all(1)
    e = np.diff(np.r_[0, (~seen).astype(np.int8), 0])
    starts, stops = np.flatnonzero(e == 1), np.flatnonzero(e == -1)
    out = {"visible": float((pred[seen] == "D").mean())}
    for lo, hi in GAP_BUCKETS:
        k = ((stops - starts) >= 25 * lo) & ((stops - starts) < 25 * hi)
        idx = np.concatenate([np.arange(a, b) for a, b in zip(starts[k], stops[k])]) if k.any() else []
        out[f"{lo}-{hi}"] = float((pred[idx] == "D").mean()) if len(idx) else None
    return out


def prepare_match(match, args, dev):
    """Write one match's feature views to disk; train views normalized float16, dev views raw float32."""
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    builder = __import__("possession.features_" + state["config"]["feature_version"], fromlist=["x"]).build_features
    prof = load_profile(json.loads(Path(args.profile).read_text()))
    cache = Path(args.cache)
    for i, h in enumerate(halves_of(match, args.sportec)):
        np.save(cache / f"{match}_{i}_y.npy", pm.label_codes(h.labels).astype(np.int64))
        for view in (args.dev_views if dev else args.train_views):
            path = cache / f"{match}_{i}_{file_name(view, args)}.npy"
            if path.exists():
                continue
            seed = SPEC[view][2] + i if dev else 10_000 + 100 * TRAIN.index(match) + 10 * i + SPEC[view][1]
            players, ball = view_tracking(h, view, prof, seed, args)
            x, names = builder(players, h.teams, ball, h.pitch, step=1)
            if names != state["names"]:
                raise ValueError("Feature schema differs from checkpoint")
            x = x[:len(h.labels)]
            x = x.astype(np.float32) if dev else normalized(x, state["mean"], state["scale"]).astype(np.float16)
            np.save(path.with_suffix(".partial.npy"), x)
            path.with_suffix(".partial.npy").replace(path)
            if dev:
                np.save(path.with_name(path.stem + "_ball.npy"), ball[:len(h.labels)])
    return match


def evaluate(model, state, dev, costs, device):
    out = {}
    for view, halves in dev.items():
        probs = [symmetric_probability(model, h["x"], state["names"], state["mean"], state["scale"], device) for h in halves]
        out[view] = {}
        for c in costs:
            preds = [viterbi_labels(p, *c)[:len(h["truth"])] for p, h in zip(probs, halves)]
            balls = [h["ball"] for h in halves]
            out[view][json.dumps(c)] = dict(score([h["truth"] for h in halves], preds, balls), dead_by_gap=dead_by_gap(preds, balls))
    return out


def select(result):
    """Pick the decoder cost maximizing mean spell F1 over the degraded dev views."""
    views = [v for v in result if v != "clean"]
    mean = lambda c: float(np.mean([result[v][c]["spell_f1"] for v in views]))
    best = max(result[views[0]], key=mean)
    keys = ("spell_f1", "macro_f1", "dead_recall", "pred_dead_share", "dead_by_gap")
    summary = {v: {k: result[v][best][k] for k in keys} for v in result}
    return best, mean(best), summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="/content/ft-cache")
    ap.add_argument("--sportec", default=str(ROOT / "data/external/sportec"))
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--native-probability", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=2031)
    ap.add_argument("--train-views", nargs="+", default=["native", "carry", "mixed"], choices=[v for v in SPEC if v != "clean"])
    ap.add_argument("--dev-views", nargs="+", default=["clean", "carry", "mixed"], choices=[v for v in SPEC if v != "native"])
    ap.add_argument("--stoppage-q", type=float, default=0.0)
    ap.add_argument("--stoppage-fraction", type=float, default=1.0)
    ap.add_argument("--max-alive-gap", type=float, default=0.0, help="seconds; 0 = uncapped")
    args = ap.parse_args()
    out, cache = Path(args.out), Path(args.cache)
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    with ProcessPoolExecutor(args.workers) as pool:
        jobs = [pool.submit(prepare_match, m, args, False) for m in TRAIN] + [pool.submit(prepare_match, "J03WPY", args, True)]
        for job in jobs:
            print("prepared", job.result(), round(time.monotonic() - started), flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    from scripts.run_local import model_from_state
    from possession.boundary import boundary_loss
    model = model_from_state(state, device)
    names, config = state["names"], state["config"]
    costs = config["development_decoder_costs"]

    native, degraded = [], []
    for m in TRAIN:
        for i in range(2):
            y = np.load(cache / f"{m}_{i}_y.npy")
            y[y == pm.X] = -100
            w = training_weights(np.array(["H", "A", "D", "X"])[np.where(y < 0, 3, y)], config.get("run_weight_power", .5))
            for view in args.train_views:
                x = np.load(cache / f"{m}_{i}_{file_name(view, args)}.npy")
                (native if view == "native" else degraded).append((x, y, w))
    truth = [np.array(["H", "A", "D", "X"])[np.load(cache / f"J03WPY_{i}_y.npy")] for i in range(2)]
    dev = {v: [dict(x=np.load(cache / f"J03WPY_{i}_{file_name(v, args)}.npy"), truth=truth[i],
                    ball=np.load(cache / f"J03WPY_{i}_{file_name(v, args)}_ball.npy"))
               for i in range(2)] for v in args.dev_views}

    params = [p for p in model.corrector.parameters()]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.lr / 10)
    mapping = torch.as_tensor([names.index("away_" + n[5:] if n.startswith("home_") else "home_" + n[5:] if n.startswith("away_") else n)
                               for n in names], device=device)
    length, batch_size = config["sequence_length"], config["batch_size"]

    history, best = [], -1.
    for epoch in range(args.epochs + 1):
        t0, losses = time.monotonic(), []
        if epoch:
            model.train()
            for _ in range(args.steps):
                bx, by, bw = [], [], []
                for _ in range(batch_size):
                    pool_ = native if rng.random() < args.native_probability else degraded
                    x, y, w = pool_[int(rng.integers(len(pool_)))]
                    a = int(rng.integers(max(1, len(x) - length + 1)))
                    bx.append(x[a:a + length].T); by.append(y[a:a + length]); bw.append(w[a:a + length])
                bx = torch.as_tensor(np.stack(bx), device=device).float()
                by = torch.as_tensor(np.stack(by), device=device)
                bw = torch.as_tensor(np.stack(bw), device=device)
                swap = torch.as_tensor(rng.random(batch_size) < .5, device=device)
                bx[swap] = bx[swap][:, mapping]
                by[swap] = torch.where(by[swap] == 0, 1, torch.where(by[swap] == 1, 0, by[swap]))
                optimizer.zero_grad(set_to_none=True)
                logits, edge = model.forward_with_boundary(bx)
                loss = (frame_loss(logits, by) * bw).sum() / bw.sum().clamp_min(1.)
                loss = loss + config.get("boundary_auxiliary_weight", .3) * boundary_loss(edge, by, model.positive_weight, 0)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.)
                optimizer.step()
                losses.append(float(loss.detach()))
            scheduler.step()
        cost, proxy, summary = select(evaluate(model, state, dev, costs, device))
        entry = dict(epoch=epoch, loss=float(np.mean(losses)) if losses else None, proxy_spell_f1=proxy,
                     decoder_costs=json.loads(cost), views=summary, seconds=round(time.monotonic() - t0, 1))
        history.append(entry)
        print(json.dumps(entry), flush=True)
        if proxy > best:
            best = proxy
            torch.save(dict(state, model=model.state_dict(), decoder_costs=json.loads(cost), validation=summary, epoch=epoch,
                            weights_kind="raw", finetune=dict(vars(args), parent=args.checkpoint, selection="J03WPY " + "+".join(v for v in args.dev_views if v != "clean"))),
                       out / "selected.pt")
        (out / "history.json").write_text(json.dumps(history, indent=1))


if __name__ == "__main__":
    main()
