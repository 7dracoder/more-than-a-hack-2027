"""Explicit, team-symmetric H/A segment edits; proposals never receive labels."""
from bisect import bisect_right
from dataclasses import dataclass
import numpy as np
from scipy.signal import find_peaks
import torch
from torch import nn
from evaluation import possession_metrics as pm


@dataclass(frozen=True, order=True)
class Edit:
    start: int
    stop: int
    target: int
    kind: str


def runs(labels):
    labels = np.asarray(labels)
    edges = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1])+1, len(labels)]
    return [(int(a), int(b), int(labels[a])) for a, b in zip(edges[:-1], edges[1:])]


def propose(probability, labels, config):
    p = np.asarray(probability)
    labels = np.asarray(labels)
    if p.shape != (len(labels), 4) or not len(labels) or not np.isfinite(p).all():
        raise ValueError('Expected finite H/A/D/boundary probabilities and one decoded half')
    if not np.isin(labels, [0, 1, 2, 3]).all():
        raise ValueError('Invalid decoded labels')
    segments = runs(labels); candidates = set()
    for i, (a, b, k) in enumerate(segments):
        if k not in (0, 1):
            continue
        if (0 < i < len(segments)-1 and b-a <= config['max_island_frames'] and
                segments[i-1][2] == segments[i+1][2] == 1-k):
            candidates.add(Edit(a, b, 1-k, 'delete'))
        if i and segments[i-1][2] == 1-k:
            left = segments[i-1][0]
            for offset in (-12, -8, -4, -2, -1, 1, 2, 4, 8, 12):
                if abs(offset) > config['max_shift_frames']:
                    continue
                edge = a+offset
                if left < edge < b:
                    candidates.add(Edit(min(a, edge), max(a, edge), 1-k if offset > 0 else k, 'shift'))
        if b-a < 5:
            continue
        peaks, _ = find_peaks(p[a:b, 1-k], distance=config['peak_separation'], prominence=.02,
                              height=config['minimum_alternative_probability'])
        for peak in peaks+a:
            for length in config['insert_lengths']:
                first = int(peak)-length//2; last = first+length
                if a < first < last < b:
                    candidates.add(Edit(first, last, 1-k, 'insert'))
    return sorted(candidates)


PHYSICAL = ('ball_seen', 'ball_speed', 'ball_fine_speed2', 'ball_fine_impulse2', 'ball_fine_turn2',
            'home_distance1', 'away_distance1', 'home_fine_relative2_top1', 'away_fine_relative2_top1',
            'home_fine_control_mass2', 'away_fine_control_mass2', 'home_last_contact', 'away_last_contact',
            'contact_age', 'home_fine_alignment2_top1', 'away_fine_alignment2_top1',
            'home_fine_control_mass2_known', 'away_fine_control_mass2_known')


def edit_features(probability, labels, matrix, names, edits):
    """Canonicalize every edit to target=home; no frame index or identity feature."""
    p = np.asarray(probability, dtype=np.float32)
    labels = np.asarray(labels)
    if matrix.shape != (len(labels), len(names)) or p.shape != (len(labels), 4):
        raise ValueError('Repair feature timeline/schema differs')
    columns = []
    for target in (0, 1):
        canonical = [n if target == 0 or not n.startswith(('home_', 'away_')) else
                     ('away_'+n[5:] if n.startswith('home_') else 'home_'+n[5:]) for n in PHYSICAL]
        columns.append([names.index(n) for n in canonical])
    entropy = -(p[:, :3]*np.log(p[:, :3].clip(1e-6))).sum(1)
    starts = np.zeros(len(labels), np.int32); stops = starts.copy()
    for a, b, _ in runs(labels):
        starts[a:b] = a; stops[a:b] = b
    rows = []
    for edit in edits:
        a, b, target = edit.start, edit.stop, edit.target
        if not 0 <= a < b <= len(labels) or target not in (0, 1) or not np.all(labels[a:b] == 1-target):
            raise ValueError('Edit must replace a contiguous opposite-team interval')
        row = [float(edit.kind == k) for k in ('shift', 'insert', 'delete')]
        row += [np.log1p((b-a)/25.)]
        for t in (max(0, a-1), a, b-1, min(len(labels)-1, b)):
            row += [np.log1p((stops[t]-starts[t])/25.), float(labels[t] == target), float(labels[t] == 2)]
        windows = ((max(0, a-25), a), (max(0, a-5), a), (a, b),
                   (b, min(len(labels), b+5)), (b, min(len(labels), b+25)))
        for lo, hi in windows:
            if hi <= lo:
                row += [0.]*9
                continue
            block = p[lo:hi]
            ratio = np.log(block[:, target].clip(1e-6))-np.log(block[:, 1-target].clip(1e-6))
            row += [float(block[:, target].mean()), float(block[:, target].max()), float(block[:, target].min()),
                    float(block[:, 1-target].mean()), float(ratio.mean()), float(entropy[lo:hi].mean()),
                    float(block[:, 3].mean()), float(block[:, 3].max()), float(block[:, 2].mean())]
        for lo, hi in ((max(0, a-12), a), (a, b), (b, min(len(labels), b+12))):
            row += (np.asarray(matrix[lo:hi, columns[target]]).mean(0).tolist() if hi > lo else [0.]*len(PHYSICAL))
        rows.append(row)
    width = 16+45+3*len(PHYSICAL)
    result = np.asarray(rows, np.float32).reshape(-1, width)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite edit features')
    return result


def choose_edits(edits, gains, threshold=0.):
    """Maximum-weight nonoverlapping edits with a one-frame context guard."""
    gains = np.asarray(gains)
    if gains.shape != (len(edits),) or not np.isfinite(gains).all() or not np.isfinite(threshold):
        raise ValueError('Expected finite scores for every edit')
    jobs = sorted([(e.stop+1, max(0, e.start-1), e, float(g)-threshold)
                   for e, g in zip(edits, gains) if g > threshold])
    ends = [j[0] for j in jobs]
    best = [0.]; back = []; take = []
    for i, (_, left, _, weight) in enumerate(jobs):
        previous = bisect_right(ends, left, hi=i)
        value = best[previous]+weight
        picked = value > best[-1]
        best.append(value if picked else best[-1]); back.append(previous); take.append(picked)
    selected = []; cursor = len(jobs)
    while cursor:
        if take[cursor-1]:
            selected.append(jobs[cursor-1][2]); cursor = back[cursor-1]
        else:
            cursor -= 1
    return sorted(selected)


def apply_edits(labels, edits):
    result = np.asarray(labels).copy()
    previous = -1
    for edit in sorted(edits):
        if edit.start < previous or not 0 <= edit.start < edit.stop <= len(result) or edit.target not in (0, 1):
            raise ValueError('Overlapping or invalid edits')
        if not np.all(result[edit.start:edit.stop] == 1-edit.target):
            raise ValueError('Edits may change H/A only and must match their original state')
        result[edit.start:edit.stop] = edit.target; previous = edit.stop
    return result


def overlap_linear(gt, pred):
    """Exact official IoUs/tie order, using disjoint sorted spell intervals."""
    out = []; i = j = 0
    while i < len(gt) and j < len(pred):
        a, b, k = gt[i]; c, d, label = pred[j]
        if k == label and max(a, c) < min(b, d):
            out.append(((min(b, d)-max(a, c))/(max(b, d)-min(a, c)), i, j))
        if b <= d:
            i += 1
        if d <= b:
            j += 1
    return sorted(out, key=lambda row: (-row[0], row[1], row[2]))


def fast_spell_score(truth, labels):
    graded = pm.valid_runs(truth)
    gt = pm.possession_spells(truth, graded)
    pred = pm.possession_spells(pm.fill_abstentions(labels, graded), graded)
    overlaps = overlap_linear(gt, pred)
    denominator = len(gt)+len(pred)
    scores = []
    for threshold in pm.SPELL_IOUS:
        tp, fp, fn = pm.match_spells(overlaps, len(gt), len(pred), threshold)
        precision = tp/float(tp+fp) if tp+fp else 0.
        recall = tp/float(tp+fn) if tp+fn else 0.
        scores.append(2.*precision*recall/(precision+recall) if precision+recall else 0.)
    return float(np.mean(scores)), denominator


def supervision(truth, labels, edits):
    truth = np.asarray(truth)
    baseline, denominator = fast_spell_score(truth, labels)
    result = []
    for edit in edits:
        if np.any(truth[max(0, edit.start-25):min(len(truth), edit.stop+25)] == pm.X):
            result.append(np.nan)
        else:
            changed, _ = fast_spell_score(truth, apply_edits(labels, [edit]))
            result.append((changed-baseline)*denominator/2.)
    return np.asarray(result, np.float32)


class RepairMLP(nn.Module):
    def __init__(self, inputs, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(inputs, hidden), nn.GELU(), nn.Linear(hidden, hidden),
                                 nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x)[:, 0]
