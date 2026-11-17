"""Prepare conservative external partial-HAD examples, excluding Metrica Game 3."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from possession.data import sha256
from scripts.research.spatial_memory.external_guard import restrict_external
from scripts.research.spatial_memory.run_external import dump


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    profile_path = ROOT / 'data/research/natural-visibility-20261114/external_visibility.json'
    denied = restrict_external(args.out, [profile_path])
    raw = ROOT / 'data/external/metrica-raw'
    # Explicitly exclude the diagnostic fixture even though it is external.
    def exclude_game3(event, values):
        if event == 'open' and values and isinstance(values[0], (str, bytes)):
            if 'sample_game_3' in str(values[0]).lower():
                raise PermissionError('Partial-HAD preparation excludes diagnostic Game 3')
    sys.addaudithook(exclude_game3)
    config_path = ROOT / 'configs/partial_had_memory_adaptation.json'
    config = json.loads(config_path.read_text())
    assert config['partial_training_matches'] == ['METRICA1', 'METRICA2']
    manifest = json.loads((raw / 'download_manifest.json').read_text())
    assert manifest['commit_date'] < '2024-01-01'
    pins = {item['path']:item['sha256'] for item in manifest['files']}
    sources = ['scripts/research/spatial_memory/prepare_partial_had_memory.py', 'possession/metrica_samples.py',
        'possession/external_visibility.py', 'possession/features_v09.py', 'possession/features_v08.py',
        'possession/spatial_memory.py', 'possession/external.py', 'possession/data.py']
    plan = dict(config=config, config_sha256=sha256(config_path),
        sources={path:sha256(ROOT / path) for path in sources}, profile_sha256=sha256(profile_path),
        raw_manifest_sha256=sha256(raw / 'download_manifest.json'), source_commit=manifest['commit'],
        source_commit_date=manifest['commit_date'], source_attribution='Metrica Sports public sample data',
        provided_match_used=False, public_labels_used=False, validation_matches_used=False)
    if (args.out / 'preparation_plan.json').exists():
        assert json.loads((args.out / 'preparation_plan.json').read_text()) == plan
    dump(args.out / 'preparation_plan.json', plan)
    profile = json.loads(profile_path.read_text())
    assert not profile['provided_match_used'] and not profile['possession_labels_used']
    from possession.metrica_samples import load_tracking, load_events, partial_labels, file_paths, controlled
    from possession.external_visibility import degrade
    from possession.features_v09 import build_features
    from possession.spatial_memory import build_features as memory_features
    reference = json.loads((ROOT / config['initial_model'] / 'fixed_protocol.json').read_text())
    import torch
    expected = torch.load(ROOT / config['initial_model'] / 'selected.pt', map_location='cpu', weights_only=False)['names']
    cache = args.out / 'partial_features'
    cache.mkdir(exist_ok=True)
    records, unique_counts = [], {key:0 for key in 'HADX'}
    for number in (1, 2):
        paths = file_paths(raw, number)
        inputs = {key:sha256(path) for key,path in paths.items()}
        for key,path in paths.items():
            assert inputs[key] == pins[str(path.relative_to(raw)).replace('\\', '/')]
        halves, roster = load_tracking(raw, number)
        events = load_events(raw, number)
        for i, half in enumerate(halves):
            truth, evidence = partial_labels(half.frames, int(half.period), events, roster)
            for key in unique_counts:
                unique_counts[key] += int((truth == key).sum())
            alignment = []
            for event in events:
                if event.period == int(half.period) and controlled(event, roster) and np.isfinite(event.xy).all():
                    k = event.start - half.frames[0]
                    if 0 <= k < len(half.ball) and np.isfinite(half.ball[k]).all():
                        alignment.append(float(np.linalg.norm(half.ball[k] - np.asarray(event.xy) * half.pitch)))
            assert len(alignment) >= 50 and np.median(alignment) <= 2. and np.percentile(alignment, 95) <= 5.
            for view in config['partial_views']:
                stem = f'METRICA{number}_{i}_{view}'
                path = cache / (stem + '.npy')
                target = cache / (stem + '_truth.npy')
                receipt = path.with_suffix('.json')
                seed = int(hashlib.sha256(('partial-had-v1/' + stem).encode()).hexdigest()[:8], 16)
                fingerprint = dict(plan=plan, inputs=inputs, match='METRICA' + str(number), half=i, view=view, seed=seed)
                if path.exists() and receipt.exists():
                    previous = json.loads(receipt.read_text())
                    assert previous['fingerprint'] == fingerprint
                    assert sha256(path) == previous['sha256'] and sha256(target) == previous['target_sha256']
                    records.append(previous)
                    continue
                started = time.monotonic()
                players, ball = half.players, half.ball
                if view != 'native':
                    players, ball = degrade(players, half.teams, ball, half.pitch, profile, seed, mode='mixed')
                x, names = build_features(players, half.teams, ball, half.pitch, step=1)
                memory, memory_names = memory_features(players, half.teams, ball, half.pitch)
                x = np.concatenate([x[::2], memory[::2]], axis=1)
                assert names + memory_names == expected and x.shape[1] == 792
                assert np.isfinite(x).all() and x.dtype == np.float32
                np.save(path, x)
                np.save(target, truth[::2])
                record = dict(fingerprint=fingerprint, sha256=sha256(path), target_sha256=sha256(target),
                    names=expected, rows=len(x), source_frame_count=len(truth), evidence=evidence,
                    alignment_metres=dict(median=float(np.median(alignment)), p95=float(np.percentile(alignment, 95))),
                    seconds=time.monotonic() - started)
                dump(receipt, record)
                records.append(record)
                print(f'Prepared {stem}: {len(x)} rows', flush=True)
                del x, memory
    for path,digest in plan['sources'].items():
        assert sha256(ROOT / path) == digest
    dump(args.out / 'partial_preparation.json', dict(complete=True, plan=plan, records=records,
        unique_native_25hz_label_counts=unique_counts, label_views_share_targets=True,
        provided_match_used=False, public_labels_used=False, validation_matches_used=False,
        unexpected_competition_access=denied))
    print(json.dumps(dict(complete=True, unique_counts=unique_counts)), flush=True)


if __name__ == '__main__':
    main()
