"""Read-only, checksum-verified external probabilities and development targets."""
import json
from pathlib import Path
import numpy as np
from possession.data import sha256
from scripts.research.spatial_memory.external_guard import restrict_external
from scripts.research.spatial_memory.train_real_broadcast_memory import STRONG, TRANSFER, VIEWS

ROOT = Path(__file__).resolve().parents[3]
ORIGIN = ROOT / 'models/coreorigin-real-broadcast-v3'
FALLBACK = ROOT / 'models/coreorigin-continuity-v4'
PROBABILITIES = ROOT / 'data/research/continuity-calibration-20261114'


def load(out):
    origin = json.loads((ORIGIN / 'selection.json').read_text())
    fallback = json.loads((FALLBACK / 'selection.json').read_text())
    config = origin['plan']['config']
    specs = []
    for match in config['strong_development_matches']:
        for half in (0,1):
            for view in VIEWS:
                specs.append(dict(domain='strong', view=view,
                    target=STRONG / 'cache' / f'{match}_{half}_{view}_truth.npy'))
    for match in config['weak_development_matches']:
        for half in (1,2):
            specs.append(dict(domain='weak', target=TRANSFER / 'features' / f'{match}_{half}_detected_weak.npz'))
    counts = {'strong':0, 'weak':0}
    for part in specs:
        domain = part['domain']
        part['probability_path'] = PROBABILITIES / 'probabilities' / f'{domain}_{counts[domain]}.npz'
        counts[domain] += 1
    allowed = [PROBABILITIES / name for name in ('probability_freeze.json','external_verification.json','selection.json')]
    allowed += [p[key] for p in specs for key in ('target','probability_path')]
    denied = restrict_external(out, allowed)
    previous = json.loads((PROBABILITIES / 'external_verification.json').read_text())
    assert previous['complete'] and not previous['provided_match_used'] and not previous['public_labels_used']
    assert not previous['unexpected_competition_access']
    assert previous['selection_sha256'] == sha256(PROBABILITIES / 'selection.json') == sha256(FALLBACK / 'selection.json')
    assert origin['complete'] and not origin['provided_match_used'] and not origin['public_labels_used']
    assert not origin['unexpected_competition_access']
    assert sha256(ORIGIN / 'selected.pt') == origin['selected_model_sha256'] == fallback['selected_model_sha256']
    freeze = json.loads((PROBABILITIES / 'probability_freeze.json').read_text())
    assert freeze['model_sha256'] == origin['selected_model_sha256']
    hashes = {str(path.relative_to(ROOT)):sha256(path) for path in allowed[:3]}
    for part in specs:
        for key in ('target','probability_path'):
            path = part[key]
            digest = sha256(path)
            expected = (origin['plan']['caches'][str(path.relative_to(ROOT))] if key == 'target'
                        else freeze['files'][path.name])
            assert digest == expected, str(path)
            hashes[str(path.relative_to(ROOT))] = digest
        if part['domain'] == 'strong':
            part['truth'] = np.load(part['target'])
        else:
            with np.load(part['target']) as data:
                part.update(truth=data['truth'], ticks=data['ticks'], seen=data['detected_ball_seen'])
        with np.load(part['probability_path']) as data:
            part.update(probability=data['probability'], gate=data['gate'])
        expected_rows = len(part['truth']) if part['domain'] == 'strong' else len(part['probability'])
        assert part['probability'].shape == (expected_rows,4) and part['gate'].shape == (expected_rows,)
    return origin, fallback, specs, hashes, denied
