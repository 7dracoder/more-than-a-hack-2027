"""Label-free degradation profile of a tracking half: how and where the ball and players go missing.

Runs identically on BAR-ATH (locally) and on DFL halves (Colab), so a simulator fitted to the
BAR-ATH profile can degrade DFL into a BAR-ATH-like validation set. Inputs are coordinates and
team membership only; no possession or ball-status label is read.

Arrays: players (n, slots, 2) metres, teams (slots,) in {0,1}, ball (n, 2) metres, NaN = missing.
"""
import json
import sys
from pathlib import Path

import numpy as np

FPS = 25.0
CROWD_R = 3.0
BINS = {
    "crowd": [1, 2, 4],             # players within CROWD_R of the ball: 0 | 1 | 2-3 | 4+
    "ball_speed": [2.0, 8.0, 15.0],  # m/s over one frame; NaN -> own bin
    "edge": [0.0, 3.0, 10.0],        # metres inside the pitch (negative = outside)
    "team_speed": [1.0, 2.0],        # median player speed, m/s over 5 frames
}


def runs(mask):
    edges = np.diff(np.r_[0, np.asarray(mask, np.int8), 0])
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def context(players, ball, pitch):
    """Per-frame context, defined on frames where the ball is visible."""
    d = np.linalg.norm(players - ball[:, None, :], axis=-1)
    crowd = np.sum(d < CROWD_R, axis=1).astype(float)
    step = np.full(len(ball), np.nan)
    step[1:] = np.linalg.norm(ball[1:] - ball[:-1], axis=1) * FPS
    edge = np.minimum.reduce([ball[:, 0], pitch[0] - ball[:, 0], ball[:, 1], pitch[1] - ball[:, 1]])
    pv = np.full(players.shape[:2], np.nan)
    pv[5:] = np.linalg.norm(players[5:] - players[:-5], axis=-1) * FPS / 5
    with np.errstate(all="ignore"):
        team_speed = np.nanmedian(np.where(pv < 12, pv, np.nan), axis=1)
    return {"crowd": crowd, "ball_speed": step, "edge": edge, "team_speed": team_speed}


def bin_index(ctx):
    """Joint context bin per frame; NaN values get their own last bin per variable."""
    idx = np.zeros(len(next(iter(ctx.values()))), np.int64)
    for name, cuts in BINS.items():
        v = ctx[name]
        b = np.digitize(np.nan_to_num(v, nan=0.0), cuts)
        b = np.where(np.isfinite(v), b, len(cuts) + 1)
        idx = idx * (len(cuts) + 2) + b
    return idx


def profile(players, teams, ball, pitch):
    pitch = np.asarray(pitch, float)
    seen = np.isfinite(ball).all(1)
    starts, stops = runs(~seen)
    gap = (stops - starts) / FPS
    ctx = context(players, ball, pitch)
    b = bin_index(ctx)
    lost = seen[:-1] & ~seen[1:]
    at = seen[:-1]
    hazard = {}
    for k in np.unique(b[:-1][at]):
        m = at & (b[:-1] == k)
        hazard[int(k)] = [int(m.sum()), int((lost & m).sum())]
    marg = {}
    for name, cuts in BINS.items():
        v = ctx[name][:-1]
        lab = np.where(np.isfinite(v), np.digitize(np.nan_to_num(v), cuts), len(cuts) + 1)
        marg[name] = {int(j): round(float(lost[at & (lab == j)].mean()), 5)
                      for j in np.unique(lab[at]) if (at & (lab == j)).sum() >= 200}
    present = np.isfinite(players).all(-1)
    frag = []
    for s in range(players.shape[1]):
        a, z = runs(present[:, s])
        frag += list((z - a) / FPS)
    frag = np.array(frag)

    def jitter(x):
        sd = x[2:] - 2 * x[1:-1] + x[:-2]
        return float(np.nanmedian(np.abs(sd)))

    q = [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    return {
        "frames": int(len(ball)),
        "ball_missing_fraction": round(float(1 - seen.mean()), 4),
        "ball_gap_frames": (stops - starts).astype(int).tolist(),
        "ball_gaps": {"count": int(len(gap)),
                      "seconds_quantiles": dict(zip(map(str, q), np.round(np.quantile(gap, q), 2).tolist())) if len(gap) else {},
                      "missing_share_in_gaps_over": {f"{t}s": round(float(gap[gap > t].sum() / max(gap.sum(), 1e-9)), 3)
                                                     for t in (1, 5, 10, 30)}},
        "loss_hazard_per_frame": round(float(lost[at].mean()), 5),
        "loss_hazard_by_context": marg,
        "loss_hazard_joint_bins": hazard,
        "objects_per_frame": {str(t): round(float(present[:, teams == t].sum(1).mean()), 2) for t in (0, 1)},
        "player_fragment_frames": (frag * FPS).astype(int).tolist(),
        "player_fragment_seconds_quantiles": dict(zip(map(str, q), np.round(np.quantile(frag, q), 2).tolist())),
        "fragments_per_minute": round(float(len(frag) / (len(ball) / FPS / 60)), 2),
        "jitter_m_median_abs_second_diff": {"ball": jitter(ball[seen.nonzero()[0]]) if seen.sum() > 3 else None,
                                            "players": float(np.nanmedian([jitter(players[:, s]) for s in range(players.shape[1])
                                                                           if present[:, s].sum() > 100]))},
    }


def load_bar_ath(root):
    sys.path.insert(0, str(root))
    from games import GAMES, asset_paths, half_video_blocks
    from possession.data import slot_teams
    cache = Path("/tmp/bar_ath_tracking.npz")
    paths = asset_paths(data_dir=str(root / "data"))
    teams = slot_teams(paths["atd_meta"])
    pitch = np.asarray(GAMES["BAR-ATH"]["pitch"], float)
    if cache.exists():
        z = np.load(cache)
        players, ball = z["players"], z["ball"]
    else:
        n = GAMES["BAR-ATH"]["n_video_frames"]
        players = np.full((n, len(teams), 2), np.nan, np.float32)
        ball = np.full((n, 2), np.nan, np.float32)
        with open(paths["atd_raw"]) as handle:
            for line in handle:
                f, p, bb = line.strip().split(":")
                i = int(f) - 1
                players[i] = np.fromstring(p.replace(";", ","), sep=",").reshape(-1, 3)[:, :2]
                ball[i] = np.fromstring(bb, sep=",")[:2]
        players *= pitch.astype(np.float32)
        ball *= pitch.astype(np.float32)
        np.savez(cache, players=players, ball=ball)
    halves = [(players[a:b + 1], ball[a:b + 1]) for _, a, b in half_video_blocks("BAR-ATH")]
    return halves, teams, pitch


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    halves, teams, pitch = load_bar_ath(root)
    out = {f"half{i + 1}": profile(p, teams, b, pitch) for i, (p, b) in enumerate(halves)}
    dest = root / "data/research/bar_ath_degradation_profile.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))
    print(json.dumps({h: {k: v for k, v in r.items() if k not in ("loss_hazard_joint_bins", "ball_gap_frames", "player_fragment_frames")}
                      for h, r in out.items()}, indent=1))
