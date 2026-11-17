"""Evaluate the pinned ball-free PathCRF checkpoint on its external test fixture.

This is a source-node H/A transfer audit, not the paper's player-edge benchmark
or a complete HAD possession model. No checkpoint/mode selection is performed.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
from possession.data import sha256
from possession.pathcrf_coordinates import player_features, windows, source_teams
from scripts.research.spatial_memory.run_external import dump, isolate, torch_setup, score_parts

PLAN = dict(match='J03WR9', trial=140, expected_fold=6,
    expected_validation_match='J03WQQ', excluded_from_published_fit=['J03WQQ', 'J03WR9'],
    published_fit_includes_our_development_match='J03WPY',
    sample_step=5, window_steps=50, stride_steps=25, batch_size=2,
    inference='Published dynamic masked CRF; previous-source node determines H/A. Outside-source nodes abstain.',
    overlap='Strict majority of all overlapping source-team votes; ties and uncovered frames abstain.',
    input='Only player coordinates, team membership and pitch dimensions; no ball/status/events/phases/truth.',
    checkpoint_selection=False, parameter_tuning=False, provided_match_used=False,
    limitation='H/A transfer audit with a coordinate preprocessing adapter. Not exact paper reproduction, an independent test of our other models, full HAD performance, or competition performance.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    denied = isolate(args.out)
    source = ROOT / 'data/external/pathcrf-source'
    checkpoint = source / 'saved/140/model/state_dict_best.pt'
    hashes = {}
    for manifest in ('source.json', 'additional_sources.json'):
        record = json.loads((source / manifest).read_text())
        if record['commit'] != '1b7c6f85a6b78c92e4543fde72ca2b86a933f997':
            raise ValueError('Unknown external model revision')
        for path, item in record['files'].items():
            if sha256(source / path) != item['sha256']:
                raise ValueError('External source or checkpoint changed')
            hashes[path] = item['sha256']
    trial = json.loads((source / 'saved/140/args.json').read_text())
    if not trial['lomocv'] or trial['cv_fold_idx'] != PLAN['expected_fold'] or trial['node_in_dim'] != 8 or trial['crf_model'] != 'dynamic_sparse_crf':
        raise ValueError('Published checkpoint split/architecture differs')
    freeze = dict(plan=PLAN, external_sources=hashes,
        adapter_sha256=sha256(ROOT / 'possession/pathcrf_coordinates.py'),
        runner_sha256=sha256(Path(__file__)), checkpoint_sha256=sha256(checkpoint))
    if (args.out / 'freeze.json').exists():
        raise ValueError('This audit is already frozen; inspect its result instead of rerunning adaptively')
    dump(args.out / 'freeze.json', freeze)
    device = torch_setup(args.device)
    import torch
    sys.path.insert(0, str(source))
    from models.utils import build_model
    model = build_model(trial, device)
    weights = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(weights, strict=True)
    del weights
    model.eval()
    from possession.external import read_sportec
    folder = ROOT / 'data/external/sportec'
    paths = [next(folder.glob('*' + kind + '*' + PLAN['match'] + '.xml')) for kind in ('matchinformation', 'positions')]
    pins = {x['name']:x['sha256'] for x in json.loads((folder / 'download_manifest.json').read_text())['files']}
    if any(sha256(p) != pins[p.name] for p in paths):
        raise ValueError('External coordinates/labels changed')
    predictions, truths, parts = [], [], []
    for i, half in enumerate(read_sportec(*paths)):
        started = time.monotonic()
        features = player_features(half.players, half.pitch)
        votes = np.zeros(((len(features) + 4) // 5, 3), np.int32)
        batches, completed = [], 0

        def consume(batch):
            nonlocal completed
            input = torch.as_tensor(np.stack([x[1] for x in batch]), device=device)
            with torch.inference_mode():
                _, emissions, embeddings = model(input)
                ids = model.valid_orig_edge_ids
                decoded = model.crf.decode(emissions.index_select(2, ids), embeddings.index_select(2, ids))
                nodes = model.comp_src[decoded].cpu().numpy()
            for (first, _, left), src in zip(batch, nodes):
                side = source_teams(src, left)
                label = np.where(side < 0, 2, side)
                np.add.at(votes, (np.arange(first, first + PLAN['window_steps']), label), 1)
            completed += len(batch)
            if completed % 100 == 0:
                print(f'PathCRF half {i + 1}: {completed} complete windows', flush=True)

        for window in windows(features, half.teams, sample_step=PLAN['sample_step'],
                window_steps=PLAN['window_steps'], stride_steps=PLAN['stride_steps']):
            batches.append(window)
            if len(batches) == PLAN['batch_size']:
                consume(batches); batches.clear()
        if batches:
            consume(batches)
        maximum = votes[:, :2].max(1)
        known = (maximum * 2 > votes.sum(1)) & (votes.sum(1) > 0)
        small = np.where(known, np.where(votes[:, 0] > votes[:, 1], 'H', 'A'), 'X')
        prediction = np.repeat(small, 5)[:len(features)]
        np.savez_compressed(args.out / f'half_{i}.npz', votes=votes, prediction=prediction)
        # No truth is accessed by the adapter or model. Freeze predictions for
        # the entire half before computing any external metrics below.
        predictions.append(prediction)
        truths.append(half.labels.copy())
        parts.append(dict(half=i, windows=completed, frames=len(features),
            covered_frames=int((prediction != 'X').sum()), seconds=round(time.monotonic() - started, 1)))
        print(json.dumps(parts[-1]), flush=True)
    t, p = np.concatenate(truths), np.concatenate(predictions)
    ha = np.isin(t, ['H', 'A']); covered = p != 'X'
    result = dict(complete=True, parts=parts, plan=PLAN,
        ha_truth_frames=int(ha.sum()), ha_covered_frames=int((ha & covered).sum()),
        ha_coverage=float((ha & covered).sum() / ha.sum()),
        ha_accuracy_including_abstentions_as_errors=float((p[ha] == t[ha]).mean()),
        ha_accuracy_on_covered_frames=float((p[ha & covered] == t[ha & covered]).mean()) if (ha & covered).any() else None,
        ha_only_spell_metrics=score_parts([np.where(np.isin(y, ['H', 'A']), y, 'X') for y in truths], predictions),
        dead_ball_performance_measured=False, full_had_model=False,
        competition_target_achieved=False, unexpected_competition_access=denied)
    dump(args.out / 'results.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
