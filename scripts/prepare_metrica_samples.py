"""Prepare audited partial-label training caches; Game 3 features contain no targets."""
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from possession.data import sha256
from possession.metrica_samples import load_tracking, load_events, partial_labels, controlled, file_paths
from possession.features_v13 import build_features
from scripts.prepare_external_v08 import corrupt_tracking
from scripts.run_local import save_json


def main():
    raw = ROOT/'data/external/metrica-raw'
    manifest = json.loads((raw/'download_manifest.json').read_text())
    for item in manifest['files']:
        if sha256(raw/item['path']) != item['sha256']:
            raise ValueError('Downloaded public sample changed')
    if manifest['commit_date'] >= '2024-01-01':
        raise ValueError('Sample provenance does not establish a distinct earlier match')
    role = json.loads((raw/'roles.json').read_text()); assert role['held_out'] == ['METRICA3']
    sources = {p: sha256(ROOT/p) for p in ('possession/metrica_samples.py', 'scripts/prepare_metrica_samples.py',
        'possession/features_v13.py', 'possession/features_v09.py', 'possession/features_v08.py',
        'possession/external.py', 'possession/data.py', 'scripts/prepare_external_v08.py')}
    report = dict(commit=manifest['commit'], commit_date=manifest['commit_date'], sources=sources,
                  training_matches=role['training'], held_out_matches=role['held_out'],
                  held_out_labels_parsed=False, label_kind='event_derived_partial',
                  provided_match_exclusion='Public sample release 2021-04-15 predates the provided 2024-2025 BAR-ATH match identified by its tactical metadata.', records=[])
    with np.load(ROOT/'data/external/features-v13/J03WMX.npz', allow_pickle=False) as reference:
        expected_names = reference['names'].tolist()
    for number in (1, 2, 3):
        started = time.monotonic(); halves, roster = load_tracking(raw, number)
        events = load_events(raw, number) if number != 3 else None
        labels = []; alignment = []
        for half in halves:
            if events is not None:
                y, evidence = partial_labels(half.frames, int(half.period), events, roster)
                for e in events:
                    if e.period == int(half.period) and controlled(e, roster) and np.isfinite(e.xy).all():
                        i = e.start-half.frames[0]
                        if np.isfinite(half.ball[i]).all():
                            alignment.append(float(np.linalg.norm(half.ball[i]-np.array(e.xy)*half.pitch)))
                labels.append((y, evidence))
            else:
                labels.append((None, dict(label_policy='Held-out event labels sealed; tracking-only cache')))
        if events is not None:
            if len(alignment) < 100 or np.median(alignment) > 2. or np.percentile(alignment, 95) > 5.:
                raise ValueError('Event-to-ball frame alignment failed')
        for corrupt in ([False, True] if number != 3 else [False]):
            directory = ROOT/('data/external/metrica-features-v13'+('-corrupt' if corrupt else ''))
            directory.mkdir(parents=True, exist_ok=True)
            path = directory/('METRICA'+str(number)+'.npz')
            if path.exists():
                meta = json.loads(path.with_suffix('.json').read_text())
                if meta['sources'] != sources or sha256(path) != meta['sha256']:
                    raise ValueError('Preserve existing cache; source/hash changed')
                report['records'].append(meta); continue
            arrays = {}; records = []; names = None
            for i, (half, (y, evidence)) in enumerate(zip(halves, labels)):
                players, ball = half.players, half.ball
                if corrupt:
                    players, ball = corrupt_tracking(players, ball, 2029+number*10+i)
                x, current = build_features(players, half.teams, ball, half.pitch, step=1)
                if current != expected_names:
                    raise ValueError('Additional-data feature schema differs from the frozen parent')
                if names is not None and names != current:
                    raise ValueError('Metrica feature schemas differ')
                if not np.isfinite(x).all() or x.dtype != np.float32:
                    raise ValueError('Invalid feature values')
                names = current; arrays['x'+str(i)] = x; arrays['frames'+str(i)] = half.frames
                if y is not None:
                    arrays['y'+str(i)] = y
                records.append(dict(half=i, frames=len(half.frames), source_first=int(half.frames[0]),
                                    source_last=int(half.frames[-1]), **evidence))
            arrays['names'] = np.array(names)
            partial = path.with_suffix('.partial.npz'); np.savez(partial, **arrays); partial.replace(path)
            inputs = {k: sha256(v) for k, v in file_paths(raw, number).items() if k != 'events' or events is not None}
            meta = dict(match='METRICA'+str(number), corrupt=corrupt, sha256=sha256(path),
                        sources=sources, source_files=inputs, source_commit=manifest['commit'],
                        targets_present=events is not None, halves=records,
                        event_alignment_metres=dict(median=float(np.median(alignment)), p95=float(np.percentile(alignment, 95))) if alignment else None)
            save_json(path.with_suffix('.json'), meta); report['records'].append(meta)
            print(json.dumps(dict(match=meta['match'], corrupt=corrupt, alignment=meta['event_alignment_metres'],
                                  frames=[r['frames'] for r in records], seconds=round(time.monotonic()-started, 2))), flush=True)
    save_json(ROOT/'data/local-v08/metrica_data_audit.json', report)


if __name__ == '__main__':
    main()
