"""Prepare real detected-only external observations for weak H/A adaptation."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from possession.data import sha256
from possession.skillcorner import load_match, resample_coordinates
from possession.features_v09 import build_features
from possession.spatial_memory import build_features as memory_features
from scripts.research.spatial_memory.external_guard import restrict_external
from scripts.research.spatial_memory.run_external import dump


def conservative_labels(labels, source_low, target_ticks, source_fps=10., target_fps=25.):
    position = np.asarray(target_ticks, np.float64) * source_fps / target_fps
    low = source_low[target_ticks]
    high = np.minimum(low + 1, len(labels) - 1)
    exact = np.isclose(position, np.rint(position), atol=1e-9, rtol=0)
    result = labels[low].copy()
    result[~exact & (labels[high] != result)] = 'X'
    if set(np.unique(result)) - set('HAX'):
        raise ValueError('Weak labels must never infer D')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/real_broadcast_memory_adaptation.json')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    denied = restrict_external(args.out)
    config = json.loads(args.config.read_text())
    original = json.loads((ROOT / 'configs/skillcorner_encoder_pretrain.json').read_text())
    if (config['weak_train_matches'] != original['train_matches'] or
            config['weak_development_matches'] != original['validation_matches'] or
            set(config['weak_train_matches']) & set(config['weak_development_matches']) or
            config['provided_match_used'] or config['public_labels_used']):
        raise ValueError('External-only data split changed')
    model_dir = ROOT / config['initial_model']
    model_manifest = json.loads((model_dir / 'model_manifest.json').read_text())
    if sha256(model_dir / 'selected.pt') != model_manifest['files']['selected.pt']:
        raise ValueError('Initial model changed')
    import torch
    state = torch.load(model_dir / 'selected.pt', map_location='cpu', weights_only=False)
    sources = ['scripts/research/spatial_memory/prepare_real_broadcast_memory.py',
        'scripts/research/spatial_memory/external_guard.py', 'possession/skillcorner.py',
        'possession/features_v08.py', 'possession/features_v09.py', 'possession/spatial_memory.py', 'possession/external.py']
    fingerprint = dict(config_sha256=sha256(args.config), config=config,
        initial_model_sha256=sha256(model_dir / 'selected.pt'), sources={p:sha256(ROOT / p) for p in sources})
    plan_path = args.out / 'weak_preparation_plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != fingerprint:
        raise ValueError('Weak preparation changed; preserve the old study and use another directory')
    dump(plan_path, fingerprint)
    cache = args.out / 'weak_features'
    cache.mkdir(exist_ok=True)
    rows = []
    for match in config['weak_train_matches']:
        folder = ROOT / 'data/external/skillcorner-raw' / match
        source = json.loads((folder / 'source.json').read_text(encoding='utf-8'))
        if sha256(folder / 'metadata.json') != source['metadata_sha256'] or sha256(folder / 'tracking.jsonl') != source['tracking_sha256']:
            raise ValueError('External raw source changed')
        metadata = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
        pitch = np.array([metadata['pitch_length'], metadata['pitch_width']], np.float32)
        halves, teams = load_match(metadata, folder / 'tracking.jsonl')
        for period in (1, 2):
            path = cache / f'{match}_{period}_detected.npy'
            target_path = path.with_name(path.stem + '_weak.npy')
            receipt_path = path.with_suffix('.json')
            expected = dict(preparation=fingerprint, source=source, match=match, period=period)
            if path.exists() and receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                if (receipt['fingerprint'] != expected or receipt['sha256'] != sha256(path) or
                        receipt['target_sha256'] != sha256(target_path)):
                    raise ValueError('Existing weak training cache changed')
                rows.append(receipt['summary'])
                print('Reused ' + path.name, flush=True)
                continue
            started = time.monotonic()
            half = halves[period]
            players, ball = half['detected']
            p, low = resample_coordinates(players + pitch / 2., speed_limit=15.)
            b, ball_low = resample_coordinates(ball + pitch / 2., speed_limit=80.)
            np.testing.assert_array_equal(low, ball_low)
            base, names = build_features(p, teams, b, pitch, step=2)
            memory, memory_names = memory_features(p, teams, b, pitch)
            matrix = np.concatenate([base, memory[::2]], axis=1)
            del base, memory, p, b
            if matrix.shape[1] != 792 or names + memory_names != state['names']:
                raise ValueError('Weak training feature schema differs from the initial model')
            target_ticks = np.arange(0, len(low), 2)
            labels = conservative_labels(half['labels'], low, target_ticks)
            assert len(labels) == len(matrix)
            partial = path.with_suffix('.partial.npy')
            np.save(partial, matrix)
            partial.replace(path)
            np.save(target_path, labels)
            summary = dict(match=match, period=period, steps=len(matrix), labels={k:int((labels == k).sum()) for k in 'HAX'},
                source_fps=10, input_step_seconds=.08, ball_missing_fraction=float((~np.isfinite(ball).all(1)).mean()))
            dump(receipt_path, dict(fingerprint=expected, names=names + memory_names,
                sha256=sha256(path), target_sha256=sha256(target_path), summary=summary,
                seconds=round(time.monotonic() - started, 1)))
            rows.append(summary)
            print(f'Prepared real weak training {path.name}: {len(matrix)} steps, {time.monotonic() - started:.1f}s', flush=True)
            del matrix
    dump(args.out / 'weak_preparation.json', dict(complete=True, fingerprint=fingerprint, rows=rows,
        provider_labels_only=True, dense_dead_ball_labels_inferred=False, validation_matches_used=False,
        provided_match_used=False, public_labels_used=False, unexpected_competition_access=denied))


if __name__ == '__main__':
    main()
