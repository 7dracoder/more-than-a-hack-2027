"""Apply the label-free out-of-bounds dead-ball rule to a candidate CSV.

Inputs: the provided ATD ball track (normalised 0-1 pitch coordinates) and our
candidate CSV. No ground truth is read. Writes the new CSV and a JSON receipt.

    python scripts/apply_dead_ball_rules.py \
        --candidate data/submissions/CoreOrigin-v08/CoreOrigin.csv \
        --out data/submissions/CoreOrigin-v09-oob/CoreOrigin.csv
"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from games import GAMES, half_video_blocks  # noqa: E402
from possession.dead_ball_rules import apply_oob, oob_dead_mask, runs_of, MARGIN_M, MIN_FRAMES  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_ball(path, n, pitch):
    ball = np.full((n, 2), np.nan, np.float64)
    with open(path) as handle:
        for i, line in enumerate(handle):
            point = np.array(line.strip().split(':')[2].split(','), dtype=np.float64) \
                if line.count(':') == 2 else np.array([])
            if point.size >= 2:
                ball[i] = point[:2]
    if i + 1 != n:
        raise ValueError('ATD row count mismatch')
    return ball * np.asarray(pitch, np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--candidate', default=str(ROOT / 'data/submissions/CoreOrigin-v08/CoreOrigin.csv'))
    ap.add_argument('--out', default=str(ROOT / 'data/submissions/CoreOrigin-v09-oob/CoreOrigin.csv'))
    ap.add_argument('--atd', default=str(ROOT / 'data/tactical/BAR-ATH_tactical_atd.txt'))
    ap.add_argument('--require-set-piece', action='store_true',
                    help='keep only runs starting 15 s before to 35 s after a Smart Tagging SET PIECES start')
    args = ap.parse_args()
    n, pitch = GAMES['BAR-ATH']['n_video_frames'], GAMES['BAR-ATH']['pitch']
    rows = Path(args.candidate).read_text().splitlines()
    if rows[0] != 'frame,label' or len(rows) != n + 1:
        raise ValueError('candidate is not a 199319-row submission')
    frames = np.array([int(r.split(',')[0]) for r in rows[1:]])
    labels = np.array([r.split(',')[1] for r in rows[1:]])
    if not np.array_equal(frames, np.arange(n)):
        raise ValueError('frame column mismatch')
    graded = np.zeros(n, bool)
    for _, start, stop in half_video_blocks('BAR-ATH'):
        graded[start:stop + 1] = True
    ball = load_ball(args.atd, n, pitch)
    if args.require_set_piece:
        sys.path.insert(0, str(ROOT / 'starter'))
        from load_smart_tagging import load as load_tags
        starts = np.array([e['start_frame'] for e in load_tags() if e['code'] == 'SET PIECES'])
        fps = 25
        for a, b in runs_of(oob_dead_mask(ball, pitch)):
            if not np.any((starts - a >= -15 * fps) & (starts - a <= 35 * fps)):
                graded[a:b] = False
    new, changed = apply_oob(labels, ball, pitch, graded)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('frame,label\n' + '\n'.join(f'{i},{l}' for i, l in enumerate(new)) + '\n')

    def spells(lab):  # maximal H or A runs, the scorer's possession definition
        return int(np.sum((np.r_[True, lab[1:] != lab[:-1]]) & np.isin(lab, ['H', 'A'])))
    receipt = dict(rule='out-of-bounds dead ball', margin_m=MARGIN_M, min_frames=MIN_FRAMES,
                   constants_selected_on='six Sportec Open DFL training matches (BallStatus precision)',
                   require_set_piece=args.require_set_piece,
                   ground_truth_read=False, candidate=args.candidate, candidate_sha256=sha(args.candidate),
                   atd_sha256=sha(args.atd), source_sha256=sha(__file__),
                   frames_changed=int(changed.sum()), runs_changed=len(runs_of(changed)),
                   predicted_possessions_before=spells(labels), predicted_possessions_after=spells(new),
                   out=str(out), out_sha256=sha(out))
    out.with_name('dead_ball_rules.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
