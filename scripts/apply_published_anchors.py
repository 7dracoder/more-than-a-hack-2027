"""Apply the published dead-ball anchor to an exported candidate.

The organisers publish that the second half is graded from frame 110808 but does
not kick off until frame 113742, and that those frames are all dead ball. A
candidate decoded from tracking alone has no reason to know that, so it usually
invents one or more possessions there. This script rewrites exactly that stretch
and leaves every other frame untouched.

It is deterministic and adds no information beyond the published boundaries, so
the same command regenerates the same file from the same candidate.

Usage:
    python scripts/apply_published_anchors.py \
        --candidate data/local-v08/run-58-pretrained-attention/candidate/CoreOrigin.csv \
        --out data/submissions/CoreOrigin-v08/CoreOrigin.csv
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from evaluation import possession_metrics as pm  # noqa: E402
from possession.anchors import (GRADED_BLOCKS, GUARD_SECONDS, SECOND_HALF_KICKOFF,  # noqa: E402
                                apply_published_dead, published_dead_intervals)

EXPECTED_ROWS = 199319


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def spells(labels):
    """Possession spells inside the graded blocks, for the before/after report."""
    out = []
    for start, stop in GRADED_BLOCKS:
        window = labels[start:stop + 1]
        breaks = np.flatnonzero(window[1:] != window[:-1]) + 1
        bounds = np.concatenate(([0], breaks, [len(window)]))
        for i in range(len(bounds) - 1):
            code = window[bounds[i]]
            if code in ('H', 'A'):
                out.append((start + int(bounds[i]), start + int(bounds[i + 1]), code))
    return out


def second_half_head(labels):
    start = GRADED_BLOCKS[1][0]
    head = labels[start:SECOND_HALF_KICKOFF]
    return {str(label): int(count) for label, count in zip(*np.unique(head, return_counts=True))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', default='data/local-v08/run-58-pretrained-attention/'
                                              'candidate/CoreOrigin.csv')
    parser.add_argument('--out', default='data/submissions/CoreOrigin-v08/CoreOrigin.csv')
    args = parser.parse_args()

    source = ROOT / args.candidate
    frame = pm.read_label_csv(source)
    problem = pm.validate_submission(frame, EXPECTED_ROWS)
    if problem:
        raise SystemExit('input candidate is not a valid submission: ' + problem)

    before = frame['label'].values
    after, changed = apply_published_dead(before)
    target = ROOT / args.out
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, 'w', newline='') as handle:
        handle.write('frame,label\n')
        handle.writelines('%d,%s\n' % (index, label) for index, label in enumerate(after))

    written = pm.read_label_csv(target)
    problem = pm.validate_submission(written, EXPECTED_ROWS)
    if problem:
        raise SystemExit('output is not a valid submission: ' + problem)
    if not np.array_equal(written['label'].values, after):
        raise SystemExit('written file does not round-trip')

    spells_before, spells_after = spells(before), spells(after)
    removed = [s for s in spells_before if not (s[1] <= published_dead_intervals()[0][0]
                                                or s[0] >= published_dead_intervals()[0][1])]
    record = dict(
        candidate=args.candidate, output=args.out,
        candidate_sha256=digest(source), output_sha256=digest(target),
        rows=len(after), frames_changed=changed,
        anchor=SECOND_HALF_KICKOFF, guard_seconds=GUARD_SECONDS,
        intervals=[[int(a), int(b)] for a, b in published_dead_intervals()],
        second_half_head_before=second_half_head(before),
        second_half_head_after=second_half_head(after),
        spells_before=len(spells_before), spells_after=len(spells_after),
        spells_removed_in_anchor=int(len(spells_before) - len(spells_after)),
        note='Published boundaries only; no ground-truth label and no tuning set is read.',
    )
    (target.parent / 'anchors.json').write_text(json.dumps(record, indent=2, sort_keys=True))

    print('rows %d, frames changed %d' % (len(after), changed))
    print('second half before kickoff: %s -> %s' % (
        record['second_half_head_before'], record['second_half_head_after']))
    print('spells %d -> %d' % (len(spells_before), len(spells_after)))
    print('output sha256', record['output_sha256'])
    print('written', args.out)


if __name__ == '__main__':
    main()
