"""Decompose the frozen development score into decoder, boundary and team parts.

Reads only the saved per-frame probabilities of the external development match
J03WPY (Sportec/DFL labels -- another game) and the unchanged official scorer.
No BAR-ATH ground truth is read, and nothing here trains or selects a BAR-ATH
model; the purpose is to say *where* the remaining spell error lives before more
compute is spent on it.

Three measurements:

1. reproduce    the recorded development score from the saved probabilities;
2. decoder      a declared transition-cost grid, to show how much of the score
                is decoder choice rather than evidence;
3. decomposition
                oracle spell boundaries with the model's own team decisions,
                and oracle team decisions with the model's own boundaries. The
                first bounds what *any* boundary detector could add, the second
                bounds what *any* team classifier could add.

Usage:
    python scripts/audit_decoder_ceiling.py \
        --cache data/local-v08/run-58-pretrained-attention/campaign-audit/probabilities.npz \
        --out data/local-v08/decoder_ceiling.json
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
import sys  # noqa: E402

sys.path.insert(0, str(ROOT))
from evaluation import possession_metrics as pm  # noqa: E402
from possession.temporal import viterbi_labels  # noqa: E402

LETTERS = np.array(['H', 'A', 'D'])
H, A, D, X = 0, 1, 2, 3
RECORDED_SCORE = 0.8570048309178743  # run-58 selection.json, fixed development only
DECLARED_COSTS = (0., 2., 1.)        # team, stoppage, boundary log-odds weight
TEAM_GRID = (0., 1., 2., 4.)
DEAD_GRID = (1., 2., 4.)
WEIGHT_GRID = (0., .5, 1., 1.5)
BIAS_GRID = (-.5, 0., .5)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(cache):
    with np.load(cache) as stored:
        return [(stored['p0'], stored['truth0']), (stored['p1'], stored['truth1'])]


def evaluate(predictions, truths):
    """Join the two halves with X exactly as the local runner does, then score."""
    truth = []
    for labels in truths:
        truth.extend([labels, np.array(['X'])])
    preds = []
    for labels in predictions:
        preds.extend([labels, np.array(['X'])])
    return pm.evaluate(pm.label_codes(np.concatenate(truth)), pm.label_codes(np.concatenate(preds)))


def summary(result):
    keys = ('spell_f1_mean', 'macro_f1', 'n_gt_spells', 'n_pred_spells')
    return {key: result[key] for key in keys}


def decode_all(halves, costs):
    team, dead, weight = costs
    return [viterbi_labels(probability, team, dead, weight) for probability, _ in halves]


def oracle_boundaries(probability, truth):
    """Keep the ground-truth spell extents; take the team from the model."""
    gt = pm.label_codes(truth)
    argmax = LETTERS[probability[:, :3].argmax(1)]
    out = LETTERS[gt].astype('<U1')
    for start, stop, code in pm.segments(gt, [(0, len(gt))]):
        if code not in (H, A):
            continue
        observed = argmax[start:stop]
        observed = observed[observed != 'D']
        if len(observed):
            values, counts = np.unique(observed, return_counts=True)
            out[start:stop] = values[counts.argmax()]
    return out


def oracle_teams(prediction, truth):
    """Keep the model's spell extents; take the majority ground-truth team."""
    gt = pm.label_codes(truth)
    codes = pm.label_codes(prediction)
    out = np.array(prediction, dtype='<U1').copy()
    for start, stop, code in pm.segments(codes, [(0, len(codes))]):
        if code not in (H, A):
            continue
        observed = gt[start:stop]
        observed = observed[observed < 2]
        if len(observed):
            values, counts = np.unique(observed, return_counts=True)
            out[start:stop] = LETTERS[values[counts.argmax()]]
    return out


def spell_rows(halves):
    rows = []
    for probability, truth in halves:
        gt = pm.label_codes(truth)
        prediction = viterbi_labels(probability, *DECLARED_COSTS)
        spells = [(a, b, k) for a, b, k in pm.segments(gt, [(0, len(gt))]) if k in (H, A)]
        predicted = [s for s in pm.segments(pm.label_codes(prediction), [(0, len(gt))]) if s[2] in (H, A)]
        overlaps = pm.spell_overlaps(spells, predicted)
        matched = pm.matched_gt_flags(overlaps, len(spells), pm.SPELL_IOU_HEADLINE)
        for index, (start, stop, code) in enumerate(spells):
            previous = next((spells[j][2] for j in range(index - 1, -1, -1)
                             if spells[j][2] in (H, A)), None)
            following = spells[index + 1][2] if index + 1 < len(spells) else None
            rows.append(dict(seconds=(stop - start) / 25.0, code=int(code),
                             before=int(gt[start - 1]) if start else X,
                             previous=previous, following=following, matched=bool(matched[index])))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', default='data/local-v08/run-58-pretrained-attention/'
                                          'campaign-audit/probabilities.npz')
    parser.add_argument('--out', default='data/local-v08/decoder_ceiling.json')
    args = parser.parse_args()
    halves = load(ROOT / args.cache)
    truths = [truth for _, truth in halves]

    record = dict(
        scope='Fixed external development only; never independent evidence.',
        cache=args.cache,
        cache_sha256=digest(ROOT / args.cache),
        scorer_sha256=digest(ROOT / 'evaluation/possession_metrics.py'),
        frames=[int(len(truth)) for _, truth in halves],
    )

    baseline = evaluate(decode_all(halves, DECLARED_COSTS), truths)
    record['reproduction'] = dict(declared_costs=list(DECLARED_COSTS), recorded=RECORDED_SCORE,
                                  **summary(baseline))
    record['reproduced_exactly'] = bool(abs(baseline['spell_f1_mean'] - RECORDED_SCORE) < 1e-12)

    grid = []
    for costs in itertools.product(TEAM_GRID, DEAD_GRID, WEIGHT_GRID, BIAS_GRID):
        result = evaluate([viterbi_labels(p, *costs) for p, _ in halves], truths)
        grid.append(dict(costs=list(costs), **summary(result)))
    grid.sort(key=lambda row: -row['spell_f1_mean'])
    record['decoder_grid'] = dict(n=len(grid), best=grid[0], worst=grid[-1],
                                  top_five=grid[:5])

    boundary = evaluate([oracle_boundaries(p, t) for p, t in halves], truths)
    teams = evaluate([oracle_teams(viterbi_labels(p, *DECLARED_COSTS), t) for p, t in halves], truths)
    record['decomposition'] = dict(
        model_as_is=summary(baseline),
        oracle_boundaries_model_teams=summary(boundary),
        model_boundaries_oracle_teams=summary(teams),
        note='Both oracles are upper bounds for their half of the problem; neither is a submission.',
    )

    rows = spell_rows(halves)
    short = [r for r in rows if r['seconds'] < 2.]
    sandwiched = [r for r in short if r['previous'] is not None
                  and r['following'] == r['previous'] and r['previous'] != r['code']]
    buckets = []
    for low, high in ((0., .5), (.5, 1.), (1., 2.), (2., 5.), (5., 10.), (10., 1e9)):
        group = [r for r in rows if low <= r['seconds'] < high]
        if group:
            buckets.append(dict(low=low, high=high, n=len(group),
                                matched=sum(r['matched'] for r in group)))
    record['spells'] = dict(
        total=len(rows), matched_at_headline=sum(r['matched'] for r in rows),
        short_under_two_seconds=len(short),
        short_matched=sum(r['matched'] for r in short),
        short_sandwiched_by_one_team=len(sandwiched),
        duration_buckets=buckets,
        note='A sandwiched short spell is a brief change of team inside the other team\'s '
             'possession -- the pattern that dominates the remaining error.',
    )

    target = ROOT / args.out
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record, indent=2, sort_keys=True))

    print('reproduced recorded development score: %s (%.10f)' % (
        record['reproduced_exactly'], baseline['spell_f1_mean']))
    print('decoder grid: %d settings, best %.6f at costs %s, worst %.6f' % (
        len(grid), grid[0]['spell_f1_mean'], grid[0]['costs'], grid[-1]['spell_f1_mean']))
    print('oracle boundaries + model teams: %.6f' % boundary['spell_f1_mean'])
    print('model boundaries + oracle teams: %.6f' % teams['spell_f1_mean'])
    print('spells: %d, matched %d; under 2 s %d (matched %d), of which %d are brief '
          'changes inside one team\'s possession' % (
              len(rows), record['spells']['matched_at_headline'], len(short),
              record['spells']['short_matched'], len(sandwiched)))
    print('written', args.out)


if __name__ == '__main__':
    main()
