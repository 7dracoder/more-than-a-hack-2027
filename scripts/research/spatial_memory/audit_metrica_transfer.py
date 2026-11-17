"""Frozen transfer diagnostic on already opened external Metrica partial labels.

Predictions are frozen before event targets are parsed. No dense spell truth,
independent test, model selection or BAR-ATH performance is claimed.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from possession.data import sha256
from scripts.research.spatial_memory.external_guard import restrict_external
from scripts.research.spatial_memory.run_external import dump, torch_setup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / 'results.json').exists():
        raise ValueError('Completed diagnostic must be preserved')
    raw = ROOT / 'data/external/metrica-raw'
    profile_path = ROOT / 'data/research/natural-visibility-20261114/external_visibility.json'
    denied = restrict_external(args.out, [profile_path])
    event_path = (raw / 'data/Sample_Game_3/Sample_Game_3_events.json').resolve()
    target_access = {'opened':False}

    def target_guard(event, values):
        if event == 'open' and values and isinstance(values[0], (str, bytes, os.PathLike)):
            if Path(os.fsdecode(values[0])).resolve() == event_path and not target_access['opened']:
                raise PermissionError('Freeze all predictions before parsing event targets')
    sys.addaudithook(target_guard)
    manifest = json.loads((raw / 'download_manifest.json').read_text())
    assert manifest['commit_date'] < '2024-01-01'
    # Event bytes were already pinned at download; no event parsing before freeze.
    for item in manifest['files']:
        if item['path'].endswith('Sample_Game_3_events.json'):
            continue
        assert sha256(raw / item['path']) == item['sha256']
    model_dirs = dict(parent=ROOT / 'models/coreorigin-video-v1', expert=ROOT / 'models/coreorigin-real-broadcast-v3')
    continuity_dir = ROOT / 'models/coreorigin-continuity-v4'
    selection = json.loads((continuity_dir / 'selection.json').read_text())
    assert selection['complete'] and not selection['provided_match_used'] and not selection['public_labels_used']
    assert sha256(model_dirs['expert'] / 'selected.pt') == selection['selected_model_sha256']
    assert sha256(model_dirs['parent'] / 'selected.pt') == selection['parent_sha256']
    source_names = ['scripts/research/spatial_memory/audit_metrica_transfer.py',
        'possession/metrica_samples.py', 'possession/external_visibility.py', 'possession/features_v25.py',
        'possession/spatial_memory.py', 'possession/uncertain_decode.py', 'possession/observation_gate.py']
    plan = dict(match='METRICA3', views=['native', 'masked', 'mixed'],
        model_sha256={key:sha256(path / 'selected.pt') for key,path in model_dirs.items()},
        protocol_sha256={key:sha256(path / 'fixed_protocol.json') for key,path in model_dirs.items()},
        continuity_selection_sha256=sha256(continuity_dir / 'selection.json'),
        source_sha256={path:sha256(ROOT / path) for path in source_names},
        raw_manifest_sha256=sha256(raw / 'download_manifest.json'),
        external_visibility_sha256=sha256(profile_path), decoder_penalty=selection['selected']['penalty'],
        provided_match_used=False, public_labels_used=False, models_retrained=False,
        model_selection_performed=False, independent_test=False,
        limitation='Previously opened external fixture, event-derived partial targets only; no dense spell truth.')
    if (args.out / 'plan.json').exists():
        assert json.loads((args.out / 'plan.json').read_text()) == plan
    dump(args.out / 'plan.json', plan)
    profile = json.loads(profile_path.read_text())
    assert not profile['provided_match_used'] and not profile['possession_labels_used']
    from possession.metrica_samples import load_tracking, load_events, partial_labels, controlled
    from possession.external_visibility import degrade
    from possession.features_v25 import build_features
    from possession.spatial_memory import build_features as memory_features
    from possession.observation_gate import mix_probabilities, observation_loss
    from possession.uncertain_decode import decode_labels
    from possession.temporal import symmetric_probability, viterbi_labels
    from scripts.run_local import selected_model
    device = torch_setup(args.device)
    parent, parent_state = selected_model(model_dirs['parent'], device)
    expert, expert_state = selected_model(model_dirs['expert'], device)
    halves, roster = load_tracking(raw, 3)
    directory = args.out / 'predictions'
    directory.mkdir(exist_ok=True)
    receipts = []
    for i, half in enumerate(halves):
        for vi, view in enumerate(plan['views']):
            path = directory / f'{i}_{view}.npz'
            receipt = path.with_suffix('.json')
            if path.exists() and receipt.exists():
                previous = json.loads(receipt.read_text())
                assert previous['plan'] == plan and previous['sha256'] == sha256(path)
                receipts.append(previous)
                continue
            started = time.monotonic()
            players, ball = half.players, half.ball
            if view != 'native':
                players, ball = degrade(players, half.teams, ball, half.pitch, profile,
                    20291100 + 10 * i + vi, mode='none' if view == 'masked' else 'mixed')
            x, names = build_features(players, half.teams, ball, half.pitch, step=1)
            assert names == parent_state['names']
            p = symmetric_probability(parent, x, names, parent_state['mean'], parent_state['scale'], device)
            memory, memory_names = memory_features(players, half.teams, ball, half.pitch)
            ex = np.concatenate([x[::2, :664], memory[::2]], axis=1)
            assert names[:664] + memory_names == expert_state['names']
            q = symmetric_probability(expert, ex, expert_state['names'], expert_state['mean'], expert_state['scale'], device)
            mixed = mix_probabilities(p, q, memory[::2], memory_names, 1.)
            gate = np.repeat(observation_loss(memory[::2], memory_names), 2)[:len(p)]
            np.savez_compressed(path, probability_parent=p, probability_adapted=mixed, gate=gate,
                parent=viterbi_labels(p, *parent_state['decoder_costs']),
                adapted=viterbi_labels(mixed, *parent_state['decoder_costs']),
                continuity=decode_labels(mixed, gate, plan['decoder_penalty'], *parent_state['decoder_costs']),
                seen=np.isfinite(ball).all(1))
            record = dict(half=i, view=view, plan=plan, sha256=sha256(path),
                rows=len(p), seconds=time.monotonic() - started)
            dump(receipt, record)
            receipts.append(record)
            del x, memory, ex, p, q, mixed
            print(f'Frozen Metrica diagnostic prediction: half {i + 1} {view}', flush=True)
    dump(args.out / 'prediction_freeze.json', dict(records=receipts, targets_parsed=False))
    target_access['opened'] = True
    event_pin = next(item['sha256'] for item in manifest['files'] if item['path'].endswith('Sample_Game_3_events.json'))
    assert sha256(event_path) == event_pin
    events = load_events(raw, 3, allow_held_out=True)
    targets, alignment = [], []
    for half in halves:
        targets.append(partial_labels(half.frames, int(half.period), events, roster)[0])
        for event in events:
            if event.period == int(half.period) and controlled(event, roster) and np.isfinite(event.xy).all():
                k = event.start - half.frames[0]
                if 0 <= k < len(half.ball) and np.isfinite(half.ball[k]).all():
                    alignment.append(float(np.linalg.norm(half.ball[k] - np.asarray(event.xy) * half.pitch)))
    assert len(alignment) >= 100 and np.median(alignment) <= 2. and np.percentile(alignment, 95) <= 5.
    from evaluation import possession_metrics as pm
    scores = {}
    for view in plan['views']:
        rows = {name:[] for name in ['parent', 'adapted', 'continuity']}
        truth = np.concatenate(targets)
        for i in range(len(halves)):
            with np.load(directory / f'{i}_{view}.npz') as data:
                for name in rows:
                    rows[name].append(data[name])
        scores[view] = {}
        for name, outputs in rows.items():
            matrix = pm.scored_confusion(pm.confusion(pm.label_codes(truth), pm.label_codes(np.concatenate(outputs))))
            scores[view][name] = dict(known_frame_accuracy=pm.accuracy(matrix), macro_HAD_f1=pm.macro_f1(matrix),
                macro_HA_f1=float(np.mean([pm.per_class_prf(matrix, k)[2] for k in (0,1)])),
                per_class={label:pm.per_class_prf(matrix, k) for k,label in enumerate('HAD')}, confusion=matrix.tolist())
    for path, digest in plan['source_sha256'].items():
        assert sha256(ROOT / path) == digest
    for key, path in model_dirs.items():
        assert sha256(path / 'selected.pt') == plan['model_sha256'][key]
        assert sha256(path / 'fixed_protocol.json') == plan['protocol_sha256'][key]
    for record in receipts:
        assert sha256(directory / f"{record['half']}_{record['view']}.npz") == record['sha256']
    all_truth = np.concatenate(targets)
    result = dict(complete=True, plan=plan, scores=scores, label_counts={k:int((all_truth == k).sum()) for k in 'HADX'},
        known_frames=int((all_truth != 'X').sum()), total_frames=len(all_truth),
        alignment_metres=dict(median=float(np.median(alignment)), p95=float(np.percentile(alignment, 95))),
        predictions_frozen_before_targets=True, unexpected_competition_access=denied,
        independent_test=False, model_selection_performed=False, full_match_spell_f1=None, competition_target_verified=False)
    dump(args.out / 'results.json', result)
    print(json.dumps(dict(complete=True, scores=scores)), flush=True)


if __name__ == '__main__':
    main()
