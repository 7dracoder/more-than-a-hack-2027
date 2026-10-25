"""Verify generated cache bytes, claim support and the sealed held-out target role."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from possession.data import sha256
from possession.metrica_samples import load_events, load_tracking, controlled
from scripts.run_local import save_json


def main():
    audit_path = ROOT/'data/local-v08/metrica_data_audit.json'
    audit = json.loads(audit_path.read_text()); records = []
    for number in (1, 2, 3):
        raw_halves, roster = load_tracking(ROOT/'data/external/metrica-raw', number)
        events = load_events(ROOT/'data/external/metrica-raw', number) if number != 3 else None
        by_index = {e.index: e for e in events} if events is not None else {}
        native_labels = []
        for corrupt in ([False, True] if number != 3 else [False]):
            path = ROOT/('data/external/metrica-features-v13'+('-corrupt' if corrupt else ''))/f'METRICA{number}.npz'
            meta = json.loads(path.with_suffix('.json').read_text())
            if sha256(path) != meta['sha256'] or any(sha256(ROOT/p) != s for p, s in meta['sources'].items()):
                raise ValueError('Cache bytes/producer differ')
            with np.load(path, allow_pickle=False) as arrays:
                if number == 3 and any(k.startswith(('y', 'truth')) for k in arrays.files):
                    raise ValueError('Sealed cache contains held-out targets')
                for i, half in enumerate(raw_halves):
                    np.testing.assert_array_equal(arrays[f'frames{i}'], half.frames)
                    x = arrays[f'x{i}']
                    if x.dtype != np.float32 or x.shape != (len(half.frames), 780) or not np.isfinite(x).all():
                        raise ValueError('Feature shape/type/finiteness audit failed')
                    del x
                    if number == 3:
                        continue
                    y = arrays[f'y{i}']; claims = np.zeros((len(y), 3), bool)
                    for claim in meta['halves'][i]['claims']:
                        a, b = claim['start']-half.frames[0], claim['stop']-half.frames[0]
                        k = 'HAD'.index(claim['label']); claims[a:b, k] = True
                        source = [by_index[j] for j in claim['events']]
                        if k == 2:
                            out, restart = source
                            if (out.kind != 'BALL OUT' or restart.kind != 'SET PIECE' or
                                out.stop != claim['start'] or restart.start != claim['stop'] or
                                not any(controlled(e, roster) and e.start == restart.start and e.team == restart.team for e in events)):
                                raise ValueError('Dead label has no explicit out/confirmed restart support')
                        elif any(not controlled(e, roster) or e.team != k for e in source):
                            raise ValueError('Possession label is not supported by a controlled pass/carry')
                    for k, label in enumerate('HAD'):
                        if np.any((y == label) & (~claims[:, k] | (claims.sum(1) != 1))):
                            raise ValueError('Known target outside a unique source claim')
                    if corrupt:
                        np.testing.assert_array_equal(y, native_labels[i])
                    else:
                        native_labels.append(y.copy())
            records.append(dict(match=f'METRICA{number}', corrupt=corrupt, sha256=meta['sha256'],
                finite_float32_features=True, exact_source_frames=True,
                targets_supported_by_source_claims=number != 3, targets_absent=number == 3))
    save_json(ROOT/'data/local-v08/metrica_cache_verification.json', dict(records=records,
        data_audit_sha256=sha256(audit_path), auditor_source_sha256=sha256(Path(__file__)),
        held_out_targets_opened=False, scope='Data derivation/alignment verification, not certification of human ground truth.'))
    print('All five caches verified; training targets supported, corruption targets unchanged, held-out labels sealed.', flush=True)


if __name__ == '__main__':
    main()
