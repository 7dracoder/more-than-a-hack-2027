"""Choose six minutes for future human validation, without inspecting model predictions.

Produces interval metadata only, never labels. Freeze models before annotation and
keep these annotations out of training until their evaluation role is retired.
"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from games import GAME, half_video_blocks
from possession.data import sha256


def main():
    source = ROOT / 'evaluation/gt/BAR-ATH_possession_gt_public.csv'
    known = pd.read_csv(source)['label'].to_numpy() != 'X'
    seed, length, embargo = 61022, 750, 300
    rng = np.random.default_rng(seed)
    rows = []
    for half, first, last in half_video_blocks(GAME):
        candidates = [a for a in range(first + embargo, last - length - embargo + 2, length + 2 * embargo)
                      if not known[max(0, a - embargo):a + length + embargo].any()]
        for a in sorted(rng.choice(candidates, size=6, replace=False).tolist()):
            rows.append(dict(half=half, first_frame=a, stop_frame=a + length,
                             video_start_seconds=a / 25, video_stop_seconds=(a + length) / 25))
    out = ROOT / 'data/validation/blind-v1'
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / 'intervals.csv', index=False)
    record = dict(seed=seed, selection='Six uniformly sampled disjoint candidate windows per half; no predictions used.',
                  public_truth_sha256=sha256(source), seconds=360, labels_created=False,
                  status='Awaiting human annotation; this is not a measured validation result.')
    (out / 'plan.json').write_text(json.dumps(record, indent=2) + '\n')
    (out / 'README.md').write_text('''# Blinded validation preparation

Twelve 30-second windows were selected without model predictions, excluding the public window and a 12-second margin. No labels have been created. These are new windows from the same match, not a new independent match.

Freeze candidate code/configuration before annotation. A reviewer should use the original video and the official EVALUATION.md definitions without seeing predictions. Record H/A/D intervals, with X for genuinely unresolvable frames; mark uncertain boundaries. Prefer a second reviewer to adjudicate uncertainty. Keep timestamps in original video frames, at 25 fps; stop_frame is exclusive.

Small hand-labelled samples for training or validation are permitted by the local RULES.md. These annotations are for validation only, never direct frame-by-frame corrections to submission output. Do not use private full-match truth. Keep match-derived materials private.

Score the frozen baseline, contacts and experimental methods once against the adjudicated labels. Report each window and pooled results. If used for model development afterward, retire this set as validation and obtain new evaluation footage. No accuracy claim follows from this sampling plan alone.
''', encoding='utf-8')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
