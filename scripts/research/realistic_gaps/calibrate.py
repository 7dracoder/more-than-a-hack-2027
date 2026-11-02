"""Choose how strongly simulated ball loss follows stoppages by matching label-free gap statistics.

  python calibrate.py --bar-ath                      # local: BAR-ATH aggregate statistics only
  python calibrate.py --reference ref.json --profile bar_ath_degradation_profile.json   # DFL sweep
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE), str(ROOT)]

import numpy as np

BUCKETS = [(0, 5), (5, 10), (10, 20), (20, 60), (60, 10_000)]


def gap_stats(players, ball, pitch):
    """Per-bucket gap counts, missing-frame shares, player-speed ratio inside gaps and edge reappearance."""
    seen = np.isfinite(ball).all(1)
    step = np.linalg.norm(players[25:] - players[:-25], axis=2)
    speed = np.r_[np.full((25, players.shape[1]), np.nan), step]
    with np.errstate(all="ignore"):
        team = np.nanmedian(np.where(speed < 12, speed, np.nan), 1)
    w, h = pitch
    e = np.diff(np.r_[0, (~seen).astype(np.int8), 0])
    starts, stops = np.flatnonzero(e == 1), np.flatnonzero(e == -1)
    seconds = (stops - starts) / 25
    visible_speed = float(np.nanmedian(team[seen]))
    out = dict(missing=float(1 - seen.mean()), gaps=len(starts), visible_speed=visible_speed, buckets={})
    for lo, hi in BUCKETS:
        k = np.flatnonzero((seconds >= lo) & (seconds < hi))
        inside = [np.nanmedian(team[starts[i]:stops[i]]) for i in k]
        after = [min(ball[stops[i], 0], w - ball[stops[i], 0], ball[stops[i], 1], h - ball[stops[i], 1])
                 for i in k if stops[i] < len(ball)]
        out["buckets"][f"{lo}-{hi}"] = dict(
            count=int(len(k)), frames=int((stops[k] - starts[k]).sum()),
            speed_ratio=float(np.nanmedian(inside) / visible_speed) if len(k) else None,
            reappear_near_edge=float(np.mean(np.array(after) < 3)) if after else None)
    return out


def pooled(stats):
    """Combine halves: counts per half, frame shares of all missing frames, count-weighted ratios."""
    total = sum(b["frames"] for s in stats for b in s["buckets"].values())
    out = dict(missing=float(np.mean([s["missing"] for s in stats])), gaps_per_half=float(np.mean([s["gaps"] for s in stats])))
    for key in stats[0]["buckets"]:
        rows = [s["buckets"][key] for s in stats]
        n = sum(r["count"] for r in rows)
        avg = lambda f: float(sum(r[f] * r["count"] for r in rows if r[f] is not None) / n) if n else None
        out[key] = dict(per_half=n / len(stats), share=sum(r["frames"] for r in rows) / total,
                        speed_ratio=avg("speed_ratio"), reappear_near_edge=avg("reappear_near_edge"))
    return out


def distance(sim, ref):
    keys = [f"{lo}-{hi}" for lo, hi in BUCKETS]
    d = sum(abs(sim[k]["share"] - ref[k]["share"]) for k in keys)
    d += sum(abs(sim[k]["per_half"] - ref[k]["per_half"]) for k in keys) / ref["gaps_per_half"]
    d += sum(abs((sim[k]["speed_ratio"] or 1) - (ref[k]["speed_ratio"] or 1)) for k in keys[1:4])
    return float(d)


def simulate_match(match, args, grid):
    from possession.external import read_sportec
    from simulate import degrade, load_profile
    prof = load_profile(json.loads(Path(args.profile).read_text()))
    folder = Path(args.sportec)
    halves = read_sportec(next(folder.glob("*matchinformation*" + match + ".xml")), next(folder.glob("*positions*" + match + ".xml")))
    rows = {}
    for q, fraction, cap in grid:
        key = f"q{q:g}_f{fraction:g}_cap{cap:g}"
        rows[key] = []
        for i, h in enumerate(halves):
            players, ball = degrade(h.players, h.teams, h.ball, h.pitch, prof, 7 + i, "carry", h.labels, q,
                                    fraction, round(25 * cap) if cap else None)
            rows[key].append(gap_stats(players, ball, h.pitch))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bar-ath", action="store_true")
    ap.add_argument("--reference", default=str(ROOT / "data/research/bar_ath_gap_reference.json"))
    ap.add_argument("--profile", default=str(ROOT / "data/research/bar_ath_degradation_profile.json"))
    ap.add_argument("--sportec", default=str(ROOT / "data/external/sportec"))
    ap.add_argument("--matches", nargs="+", default=["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WQQ", "J03WR9"])
    ap.add_argument("--q", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.7, 0.85, 1.0])
    ap.add_argument("--fraction", nargs="+", type=float, default=[1.0], help="minimum hidden share of a stoppage")
    ap.add_argument("--cap", nargs="+", type=float, default=[0.0], help="max play-time gap seconds; 0 = uncapped")
    ap.add_argument("--out", default=str(ROOT / "data/research/calibration.json"))
    args = ap.parse_args()
    if args.bar_ath:
        from degradation_profile import load_bar_ath
        halves, teams, pitch = load_bar_ath(ROOT)
        ref = pooled([gap_stats(p, b, pitch) for p, b in halves])
        Path(args.reference).write_text(json.dumps(ref, indent=1))
        print(json.dumps(ref, indent=1))
        return
    ref = json.loads(Path(args.reference).read_text())
    grid = [(q, f, c) for q in args.q for f in args.fraction for c in args.cap]
    with ProcessPoolExecutor(len(args.matches)) as pool:
        parts = list(pool.map(simulate_match, args.matches, [args] * len(args.matches), [grid] * len(args.matches)))
    result = {"reference": ref}
    for key in parts[0]:
        sim = pooled([s for part in parts for s in part[key]])
        result[key] = dict(sim, distance=distance(sim, ref))
        print(key, round(result[key]["distance"], 3), json.dumps(sim), flush=True)
    Path(args.out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
