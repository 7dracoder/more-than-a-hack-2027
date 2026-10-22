"""v0.6 fixed variants on top of the v0.5 team-only + out-of-play pipeline.

Four ideas, each with constants fixed here before any scoring:

1. Contact-gated team switches (time-varying transition costs). Possession changes at a contact
   by the new team (Vidal-Codina et al. 2022), so a switch to team T costs GATE_LOW_COST in a
   5-frame bin where a T player was within 1 m of the ball and nearer than any opponent (or in a
   neighbouring bin), and the unchanged v0.3 cost elsewhere. Switching is never made harder.
2. Live-only team model. A binary H-vs-A classifier trained only on live labelled frames, so its
   capacity is not spent on dead balls. Same booster settings and team-swap symmetry as v0.3.
3. Bagging. The classifier samples every 5th frame. Five models, one per sampling offset, are
   averaged. This only reduces variance; no setting changes.
4. Collective acceleration for live/dead (Bauer et al. 2022). Players brake together at a whistle
   and accelerate at a restart, so the dead-ball model gets future-minus-past differences of the
   team speed distribution at 1, 3 and 6 s, plus 10 s context. These features are used for the
   live/dead decision only.

Nothing here reads labels except the classifier fitting, which uses the same training masks as
every other method.
"""

import hashlib

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from games import GAME, GAMES, half_video_blocks, anchor_video_frame
from possession.champion import LETTERS, complete_runs, owner_with_restarts
from possession.decode import infer
from possession.event_features import _window

GATE_LOW_COST = 3.0
GATE_DILATE_BINS = 1
BAG_OFFSETS = (0, 1, 2, 3, 4)


# ---------------------------------------------------------------- features (live/dead only)
def acceleration_features(arrays):
    """Symmetric collective speed-change features; zero outside the halves."""
    n = len(arrays['ball'])
    q = arrays['player_speed_quantiles']
    sources = {'s50': q[:, 1], 's80': q[:, 2], 'run': arrays['running_fraction'], 'still': arrays['stationary']}
    columns = {}
    for _, start, last in half_video_blocks(GAME):
        stop = last + 1
        for key, values in sources.items():
            v = np.asarray(values[start:stop], dtype=float)
            for w in (25, 75, 150):
                name = f'accel_{key}_{w}'
                columns.setdefault(name, np.zeros(n, dtype=np.float32))
                columns[name][start:stop] = _window(v, w, 'future') - _window(v, w, 'past')
            name = f'context_{key}_250'
            columns.setdefault(name, np.zeros(n, dtype=np.float32))
            columns[name][start:stop] = _window(v, 250, 'centre')
    names = list(columns)
    return np.column_stack([np.nan_to_num(columns[k]) for k in names]), names


# ---------------------------------------------------------------- fitting
def _rows(labels, mask, stride, offset):
    rows, weights = [], []
    for a, b, _ in complete_runs(np.where(mask, labels, 'X')):
        candidates = np.arange(a + offset, b, stride)
        candidates = candidates[mask[candidates]]
        if len(candidates):
            rows.extend(candidates.tolist())
            weights.extend(np.full(len(candidates), 1 / len(candidates)))
    rows, weights = np.asarray(rows, dtype=int), np.asarray(weights)
    return rows, weights * len(weights) / weights.sum()


def _booster(config):
    return HistGradientBoostingClassifier(max_iter=config['classifier_iterations'],
        learning_rate=config['classifier_learning_rate'], max_leaf_nodes=config['classifier_max_leaf_nodes'],
        l2_regularization=config['classifier_l2'], min_samples_leaf=config['classifier_min_samples_leaf'],
        early_stopping=False, random_state=config['seed'])


def fit_three_class(matrix, names, labels, mask, config, swap, offset=0):
    """Identical to champion.fit_classifier when offset == 0."""
    rows, weights = _rows(labels, mask, config['classifier_training_stride'], offset)
    y = np.array([{'H': 0, 'A': 1, 'D': 2}[v] for v in labels[rows]])
    if set(y) != {0, 1, 2}:
        raise ValueError('Each training set needs H, A and D')
    x = np.concatenate([matrix[rows], swap(matrix[rows], names)])
    model = _booster(config)
    with threadpool_limits(limits=2):
        model.fit(x, np.r_[y, np.array([1, 0, 2])[y]], sample_weight=np.r_[weights, weights])
    return model


def three_class_probability(model, matrix, names, swap):
    with threadpool_limits(limits=2):
        return (model.predict_proba(matrix) + model.predict_proba(swap(matrix, names))[:, [1, 0, 2]]) / 2


def fit_live_team(matrix, names, labels, mask, config, swap, offset=0):
    """Binary H-vs-A model on live labelled frames only (dead frames excluded from training)."""
    live = mask & np.isin(labels, ['H', 'A'])
    rows, weights = _rows(labels, live, config['classifier_training_stride'], offset)
    y = (labels[rows] == 'A').astype(int)
    if set(y) != {0, 1}:
        raise ValueError('Live team training needs both H and A')
    x = np.concatenate([matrix[rows], swap(matrix[rows], names)])
    model = _booster(config)
    with threadpool_limits(limits=2):
        model.fit(x, np.r_[y, 1 - y], sample_weight=np.r_[weights, weights])
    return model


def live_team_probability(model, matrix, names, swap):
    """(n, 2) P(H), P(A) given live; exactly team-symmetric."""
    with threadpool_limits(limits=2):
        p = (model.predict_proba(matrix) + model.predict_proba(swap(matrix, names))[:, ::-1]) / 2
    return p


def mask_digest(mask):
    return hashlib.sha256(np.asarray(mask, dtype=bool).tobytes()).hexdigest()


# ---------------------------------------------------------------- decoding
def gated_viterbi(scores, cost_to_home, cost_to_away):
    """Two-state Viterbi with per-step switch costs. State 0 = H, 1 = A.

    cost_to_home[t] is charged for an A->H switch entering step t, cost_to_away[t] for H->A.
    With constant costs this matches possession.decode.infer exactly (same tie-breaking).
    """
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    back = np.zeros((n, 2), dtype=np.int8)
    best = scores[0].copy()
    for t in range(1, n):
        to_home = np.array([best[0], best[1] - cost_to_home[t]])
        to_away = np.array([best[0] - cost_to_away[t], best[1]])
        back[t, 0], back[t, 1] = np.argmax(to_home), np.argmax(to_away)
        best = np.array([to_home[back[t, 0]], to_away[back[t, 1]]]) + scores[t]
        best -= best.max()
    path = np.empty(n, dtype=np.int8)
    path[-1] = np.argmax(best)
    for t in range(n - 2, -1, -1):
        path[t] = back[t + 1, path[t + 1]]
    return path


def contact_gates(contacts, start, stop, step, dilate=GATE_DILATE_BINS):
    """Per-bin booleans: a home / away 1 m contact in this bin or a neighbouring one."""
    c = np.asarray(contacts[start:stop])
    begins = np.arange(0, len(c), step)
    home = np.logical_or.reduceat(c == 1, begins)
    away = np.logical_or.reduceat(c == -1, begins)
    out = []
    for g in (home, away):
        d = g.copy()
        for k in range(1, dilate + 1):
            d[k:] |= g[:-k]
            d[:-k] |= g[k:]
        out.append(d)
    return out


def decode(arrays, config, p_dead, team_log, contacts=None, low_cost=GATE_LOW_COST):
    """decode_factorized (owner reset at stoppages, no history samples) with optional gating.

    With contacts=None this reproduces champion.decode_factorized labels exactly.
    """
    n = len(arrays['ball'])
    labels = np.full(n, 'X', dtype='<U1')
    step, high = config['step'], config['team_switch_cost']
    for half, start, last in half_video_blocks(GAME):
        stop = last + 1
        dead = np.clip(p_dead[start:stop], .001, .999)
        alive_score = np.log(np.column_stack([1 - dead, dead]))
        guard = config.get('kickoff_prior_guard_frames')
        if guard is not None:
            anchor = anchor_video_frame(GAMES[GAME]['kickoff' if half == 1 else 'kickoff_2nd'])
            prefix = max(0, min(stop - start, anchor - guard - start))
            alive_score[:prefix] = [-50., 0.]

        def reduce(values):
            begins = np.arange(0, len(values), step)
            sizes = np.minimum(step, len(values) - begins)
            return np.add.reduceat(values, begins, axis=0) / sizes[:, None]
        cost = config['stoppage_switch_cost']
        binary, _, _ = infer(reduce(alive_score), np.array([[0., -cost], [-cost, 0.]]), samples=0, seed=config['seed'] + half)
        team = reduce(team_log[start:stop])
        if contacts is None:
            owner, _ = owner_with_restarts(team, binary, high)
        else:
            home_gate, away_gate = contact_gates(contacts, start, stop, step)
            to_home = np.where(home_gate, low_cost, high)
            to_away = np.where(away_gate, low_cost, high)
            owner = np.zeros(len(team), dtype=np.int8)
            live = np.asarray(binary) == 0
            edges = np.flatnonzero(np.diff(np.r_[False, live, False]))
            for first, end in zip(edges[::2], edges[1::2]):
                owner[first:end] = gated_viterbi(team[first:end], to_home[first:end], to_away[first:end])
        expand = lambda values: np.repeat(values, step, axis=0)[:stop - start]
        labels[start:stop] = LETTERS[np.where(expand(binary) == 1, 2, expand(owner))]
    return labels
