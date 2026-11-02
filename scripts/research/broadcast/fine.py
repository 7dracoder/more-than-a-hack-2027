"""Exact per-half broadcast->tactical offsets and per-tactical-frame broadcast signals (label-free)."""
import numpy as np

from load import load

HALVES = {0: (0, 85000, 4015), 1: (86000, 164488, 26920)}


def frame_cost(bp, bt, tp, tt, b_idx, t_idx, clip=5.0):
    """Team-aware one-sided chamfer for paired frames (broadcast b_idx[i] vs tactical t_idx[i])."""
    total = np.zeros(len(b_idx))
    count = np.zeros(len(b_idx))
    for team in (0, 1):
        q, c = bp[b_idx][:, bt == team], tp[t_idx][:, tt == team]
        d = np.linalg.norm(q[:, :, None, :] - c[:, None, :, :], axis=-1)
        d = np.where(np.isfinite(d), d, np.inf).min(-1)
        ok = np.isfinite(q).all(-1)
        total += np.where(ok, np.minimum(d, clip), 0).sum(1)
        count += ok.sum(1)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan), count


if __name__ == "__main__":
    tp, tb, tt, _, _ = load("tactical", "/tmp/mtah_data")
    bp, bb, bt, _, bv = load("broadcast", "/tmp/mtah_data")
    offsets = {}
    for h, (a, z, guess) in HALVES.items():
        b = np.arange(a, z, 7)
        b = b[bv[b] & (np.isfinite(bp[b]).all(-1).sum(1) >= 8)]
        scores = {}
        for off in range(guess - 12, guess + 13):
            keep = b + off < len(tp)
            c, _ = frame_cost(bp, bt, tp, tt, b[keep], b[keep] + off)
            scores[off] = np.nanmedian(c)
        offsets[h] = min(scores, key=scores.get)
        print("half", h, "offset", offsets[h], {k: round(v, 3) for k, v in sorted(scores.items()) if abs(k - offsets[h]) <= 3})

    n = len(tp)
    signal = dict(covered=np.zeros(n, bool), valid=np.zeros(n, bool), cost=np.full(n, np.nan),
                  players=np.zeros(n, np.int16), ball=np.full((n, 2), np.nan, np.float32))
    for h, (a, z, _) in HALVES.items():
        b = np.arange(a, z)
        t = b + offsets[h]
        keep = (t >= 0) & (t < n)
        b, t = b[keep], t[keep]
        signal["covered"][t] = True
        signal["valid"][t] = bv[b]
        for s in range(0, len(b), 20000):
            c, k = frame_cost(bp, bt, tp, tt, b[s:s + 20000], t[s:s + 20000])
            signal["cost"][t[s:s + 20000]] = c
            signal["players"][t[s:s + 20000]] = k
        signal["ball"][t] = bb[b]
    np.savez("/tmp/bcast_signal.npz", offsets=np.array([offsets[0], offsets[1]]), **signal)
    print("saved /tmp/bcast_signal.npz")
