"""Coarse broadcast->tactical matching: one-sided, team-aware chamfer of visible broadcast players."""
import numpy as np

from load import load


def chamfer(query, qteams, cand, cteams, clip=5.0):
    """Mean over query players of the distance to the nearest same-team candidate player (clipped)."""
    cost = np.zeros(len(cand))
    for team in (0, 1):
        q = query[qteams == team]
        if not len(q):
            continue
        c = cand[:, cteams == team]
        d = np.linalg.norm(c[:, None, :, :] - q[None, :, None, :], axis=-1)
        d = np.where(np.isfinite(d), d, np.inf).min(-1)
        cost += np.minimum(d, clip).sum(1)
    return cost / len(query)


if __name__ == "__main__":
    tp, tb, tt, _, _ = load("tactical", "/tmp/mtah_data")
    bp, bb, bt, _, bv = load("broadcast", "/tmp/mtah_data")
    stride = 5
    cand = tp[::stride]
    rows = []
    for f in range(0, len(bp), 250):
        seen = np.isfinite(bp[f]).all(1)
        if not bv[f] or seen.sum() < 10:
            continue
        cost = chamfer(bp[f][seen], bt[seen], cand, tt)
        best = int(cost.argmin())
        second = np.min(np.r_[cost[:max(0, best - 50)], cost[best + 50:]])
        rows.append((f, best * stride, cost[best], second))
    rows = np.array(rows)
    np.save("/tmp/bcast_coarse.npy", rows)
    for f, t, c, s in rows[::4]:
        print(int(f), int(t), int(t - f), round(c, 2), round(s, 2))
