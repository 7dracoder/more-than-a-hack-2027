"""Unit-correct motion model, factorized decoding and symmetric CPU classifier."""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits

from games import GAME, GAMES, half_video_blocks, anchor_video_frame
from possession.data import rolling_mean
from possession.decode import infer
from possession.metric_features import swap_teams

LETTERS = np.array(['H', 'A', 'D'])


def active_mask(n):
    mask = np.zeros(n, dtype=bool)
    for _, start, last in half_video_blocks(GAME):
        mask[start:last + 1] = True
    return mask


def fit_motion(matrix, names, config):
    selected = [names.index(key) for key in ('speed50_50', 'speed80_50', 'stationary_50', 'running_50', 'ball_seen_50')]
    rows = np.flatnonzero(active_mask(len(matrix)))[::config['gmm_training_stride']]
    observation = names.index('speed_observed_50') if 'speed_observed_50' in names else None
    if observation is not None:
        rows = rows[matrix[rows, observation] > 0]
    values = matrix[rows][:, selected]
    center = np.median(values, axis=0)
    scale = np.maximum(.2, np.quantile(values, .75, axis=0) - np.quantile(values, .25, axis=0))
    standardized = (values - center) / scale
    with threadpool_limits(limits=2):
        model = GaussianMixture(n_components=config['gmm_components'], covariance_type=config['gmm_covariance_type'],
                                reg_covar=config['gmm_regularization'], n_init=1,
                                random_state=config['seed'], max_iter=150).fit(standardized)
    physical_means = model.means_ * scale + center
    dead = int(np.argmin(physical_means[:, 0]))
    audit = {'kind': 'unlabelled_two_component_motion_gmm', 'features': [names[i] for i in selected],
             'physical_component_means': physical_means.tolist(), 'dead_component': dead,
             'component_weights': model.weights_.tolist(), 'converged': bool(model.converged_),
             'n_training_rows': len(rows), 'ground_truth_used': False,
             'identification_rule': 'Lower median-player-speed component is the candidate stoppage component; not true state supervision.'}
    return {'model': model, 'selected': selected, 'center': center, 'scale': scale, 'dead': dead,
            'observation': observation, 'audit': audit}


def motion_probability(matrix, fitted):
    values = (matrix[:, fitted['selected']] - fitted['center']) / fitted['scale']
    with threadpool_limits(limits=2):
        probability = fitted['model'].predict_proba(values)[:, fitted['dead']].clip(.01, .99)
    if fitted.get('observation') is not None:
        probability = np.where(matrix[:, fitted['observation']] > 0, probability, .5)
    return probability


def team_scores(arrays, config, use_phase=True):
    nearest = np.nan_to_num(arrays['nearest'], nan=30.)
    seen = np.isfinite(arrays['ball']).all(1)
    margin = (nearest[:, 1] - nearest[:, 0]) / (nearest.sum(1) + .5)
    reliability = np.exp(-np.maximum(0., nearest.min(1) - 1.) / 4.) * seen
    team = config['geometry_weight'] * np.clip(margin * 4, -3, 3) * reliability
    if use_phase:
        team += .5 * config['tag_weight'] * arrays['phase_support']
        team += .35 * arrays['restart_owner']
    return np.column_stack([team / 2, -team / 2])


def owner_with_restarts(scores, live_dead, cost, seed=0, sample=False):
    """Infer team independently within each live run; dead evidence has no owner.

    Owner marginals are conditional on the supplied live/dead segmentation.
    Each sampled live/dead history receives its own conditional owner sample.
    """
    owner = np.zeros(len(scores), dtype=np.int8)
    support = np.full((len(scores), 2), .5)
    live = np.asarray(live_dead) == 0
    edges = np.flatnonzero(np.diff(np.r_[False, live, False]))
    for i, (first, stop) in enumerate(zip(edges[::2], edges[1::2])):
        path, probability, draws = infer(scores[first:stop], np.array([[0., -cost], [-cost, 0.]]),
                                         samples=int(sample), seed=seed+i)
        owner[first:stop] = draws[0] if sample else path
        support[first:stop] = probability
    return owner, support


def decode_factorized(arrays, config, p_dead, conditional_team=None, use_phase=True):
    """Decode live/dead independently of the division of live mass into H/A.

    Joint argmax otherwise makes an uncertain team divide live support between
    two labels, allowing D to win with less than 50% binary dead support.
    """
    n = len(arrays['ball'])
    labels = np.full(n, 'X', dtype='<U1')
    posterior = np.zeros((n, 3), dtype=np.float32)
    histories = np.full((config['history_samples'], n), 3, dtype=np.int8)
    team = team_scores(arrays, config, use_phase) if conditional_team is None else conditional_team
    for half, start, last in half_video_blocks(GAME):
        stop = last + 1
        dead = np.clip(p_dead[start:stop], .001, .999)
        alive_score = np.log(np.column_stack([1 - dead, dead]))
        guard = config.get('kickoff_prior_guard_frames')
        if guard is not None:
            anchor_name = 'kickoff' if half == 1 else 'kickoff_2nd'
            anchor = anchor_video_frame(GAMES[GAME][anchor_name])
            prefix = max(0, min(stop-start, anchor - guard - start))
            # Public match metadata, not possession truth. Stop one second short
            # of the observed kickoff anchor to avoid assuming exact kick timing.
            alive_score[:prefix] = [-50., 0.]
        def reduce(values):
            begins = np.arange(0, len(values), config['step'])
            sizes = np.minimum(config['step'], len(values) - begins)
            return np.add.reduceat(values, begins, axis=0) / sizes[:, None]
        binary, alive_p, alive_paths = infer(reduce(alive_score),
            np.array([[0., -config['stoppage_switch_cost']], [-config['stoppage_switch_cost'], 0.]]),
            samples=config['history_samples'], seed=config['seed'] + half)
        team_coarse = reduce(team[start:stop])
        if config.get('reset_owner_at_stoppage', False):
            owner, owner_p = owner_with_restarts(team_coarse, binary, config['team_switch_cost'])
            owner_paths = np.array([owner_with_restarts(team_coarse, path, config['team_switch_cost'],
                                    config['seed']+1000*half+k, sample=True)[0]
                                    for k, path in enumerate(alive_paths)], dtype=np.int8)
        else:
            owner, owner_p, owner_paths = infer(team_coarse,
                np.array([[0., -config['team_switch_cost']], [-config['team_switch_cost'], 0.]]),
                samples=config['history_samples'], seed=config['seed'] + 10 + half)
        expand = lambda values: np.repeat(values, config['step'], axis=0)[:stop-start]
        labels[start:stop] = LETTERS[np.where(expand(binary) == 1, 2, expand(owner))]
        ap, op = expand(alive_p), expand(owner_p)
        posterior[start:stop] = np.column_stack([ap[:, 0] * op[:, 0], ap[:, 0] * op[:, 1], ap[:, 1]])
        for i in range(config['history_samples']):
            histories[i, start:stop] = np.where(expand(alive_paths[i]) == 1, 2, expand(owner_paths[i]))
    return labels, posterior, histories


def complete_runs(labels):
    edges = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
    return [(int(a), int(b), str(labels[a])) for a, b in zip(edges[:-1], edges[1:]) if labels[a] != 'X']


def purged_folds(labels, n_folds=4, embargo=250, context=125):
    """Whole-run contiguous folds; purge entire neighbouring runs from training."""
    if embargo < 2 * context:
        raise ValueError('Embargo shorter than twice feature context')
    runs = complete_runs(labels)
    if len(runs) < n_folds:
        raise ValueError('Too few labelled runs')
    if any(runs[i][1] != runs[i+1][0] for i in range(len(runs)-1)):
        raise ValueError('This development protocol expects one contiguous public window')
    starts = np.array([r[0] for r in runs] + [runs[-1][1]])
    cuts = [0]
    for k in range(1, n_folds):
        available = np.arange(cuts[-1] + 1, len(runs) - (n_folds-k) + 1)
        target = starts[0] + (starts[-1]-starts[0]) * k/n_folds
        cuts.append(int(available[np.argmin(np.abs(starts[available]-target))]))
    cuts.append(len(runs))
    result = []
    for fold, (a, b) in enumerate(zip(cuts[:-1], cuts[1:])):
        low, high = runs[a][0], runs[b-1][1]
        test = np.zeros(len(labels), dtype=bool)
        test[low:high] = True
        train = np.zeros(len(labels), dtype=bool)
        for first, stop, _ in runs:
            if stop <= low-embargo or first >= high+embargo:
                train[first:stop] = True
        result.append((train, test, {'fold': fold, 'test_first': low, 'test_stop': high,
                                     'train_frames': int(train.sum()), 'test_frames': int(test.sum()),
                                     'embargo_frames': embargo, 'complete_runs_only': True}))
    return result


def training_rows(labels, mask, stride):
    rows, weights = [], []
    if not np.isin(labels[mask], LETTERS).all():
        raise ValueError('Training mask contains unlabelled/invalid frames')
    # Every complete state run contributes equal total weight; no framewise split.
    for a, b, _ in complete_runs(np.where(mask, labels, 'X')):
        candidates = np.arange(a, b, stride)
        candidates = candidates[mask[candidates]]
        if len(candidates):
            rows.extend(candidates.tolist())
            weights.extend(np.full(len(candidates), 1/len(candidates)))
    rows, weights = np.asarray(rows, dtype=int), np.asarray(weights)
    weights *= len(weights) / weights.sum()
    return rows, weights


def fixed_time_folds(valid, n_folds=4, embargo=300):
    """Equal-duration folds using only labelled-window availability, never states."""
    ids = np.flatnonzero(valid)
    if len(ids) < n_folds or (np.diff(ids) != 1).any():
        raise ValueError('Expected one contiguous labelled window with enough frames')
    cuts = np.linspace(ids[0], ids[-1]+1, n_folds+1, dtype=int)
    frames = np.arange(len(valid))
    for fold, (first, stop) in enumerate(zip(cuts[:-1], cuts[1:])):
        test = (frames >= first) & (frames < stop)
        train = valid & ((frames < first-embargo) | (frames >= stop+embargo))
        yield train, test, {'fold': fold, 'test_first': int(first), 'test_stop': int(stop),
                            'embargo_frames': embargo, 'boundary_rule': 'Equal frame durations; no possession-state boundaries used'}


def fit_classifier(matrix, names, labels, mask, config, swap=swap_teams):
    rows, weights = training_rows(labels, mask, config['classifier_training_stride'])
    y = np.array([{'H': 0, 'A': 1, 'D': 2}[v] for v in labels[rows]])
    if set(y) != {0, 1, 2}:
        raise ValueError('Each development training set needs H, A and D')
    x = np.concatenate([matrix[rows], swap(matrix[rows], names)])
    y = np.r_[y, np.array([1, 0, 2])[y]]
    weights = np.r_[weights, weights]
    model = HistGradientBoostingClassifier(max_iter=config['classifier_iterations'],
        learning_rate=config['classifier_learning_rate'], max_leaf_nodes=config['classifier_max_leaf_nodes'],
        l2_regularization=config['classifier_l2'], min_samples_leaf=config['classifier_min_samples_leaf'],
        early_stopping=False, random_state=config['seed'])
    with threadpool_limits(limits=2):
        model.fit(x, y, sample_weight=weights)
    audit = {'kind': 'symmetric_hist_gradient_boosting', 'sampled_training_rows_before_augmentation': len(rows),
             'n_training_frames': int(mask.sum()), 'feature_names': names,
             'training_mask_sha256': __import__('hashlib').sha256(mask.astype(np.uint8).tobytes()).hexdigest(),
             'symmetry': 'Team-swap augmentation and exact probability symmetrization at prediction',
             'weighting': 'Equal total weight per retained state run',
             'warning': 'Tiny same-match public supervision; no independent-match validation'}
    return model, audit


def classifier_probability(model, matrix, names, swap=swap_teams):
    with threadpool_limits(limits=2):
        original = model.predict_proba(matrix)
        swapped = model.predict_proba(swap(matrix, names))[:, [1, 0, 2]]
    return (original + swapped) / 2
