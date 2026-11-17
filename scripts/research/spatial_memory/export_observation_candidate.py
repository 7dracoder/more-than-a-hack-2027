"""Apply the externally frozen observation-loss gate to local supplied inputs."""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from games import GAMES, asset_paths, half_video_blocks
from possession.data import sha256
from possession.observation_gate import mix_probabilities
from scripts.research.spatial_memory.export_natural_candidate import inference_guard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--ball', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=ROOT / 'data')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    paths = asset_paths(data_dir=str(args.data))
    model_files = [args.model / name for name in ('selected.pt', 'fixed_protocol.json', 'selection.json')]
    allowed = model_files + [args.ball, args.ball.with_suffix('.json')] + [paths[k] for k in ('atd_raw', 'atd_meta', 'smart_tagging')]
    denied = inference_guard(args.out, allowed)
    selection = json.loads((args.model / 'selection.json').read_text())
    if (not selection['complete'] or selection['selected']['kind'] != 'observation_loss' or
            selection['provided_match_used'] or selection['public_labels_used'] or selection['unexpected_competition_access']):
        raise ValueError('Need a frozen isolated external observation-loss selection')
    if sha256(args.model / 'selected.pt') != selection['selected_model_sha256']:
        raise ValueError('Selected expert changed')
    for name, digest in selection['plan']['sources'].items():
        if sha256(ROOT / name) != digest:
            raise ValueError('Gate-selection source changed')
    parent_dir = ROOT / 'models/coreorigin-video-v1'
    if (sha256(parent_dir / 'selected.pt') != selection['parent_sha256'] or
            sha256(parent_dir / 'fixed_protocol.json') != selection['parent_protocol_sha256']):
        raise ValueError('Frozen parent member changed')
    sources = ['scripts/research/spatial_memory/export_observation_candidate.py',
        'scripts/research/spatial_memory/export_natural_candidate.py', 'possession/observation_gate.py',
        'possession/external.py', 'possession/data.py', 'scripts/run_local.py', 'games.py',
        'possession/inference_cache.py', 'possession/features_v08.py', 'possession/features_v09.py',
        'possession/features_v13.py', 'possession/features_v25.py', 'possession/spatial_memory.py',
        'possession/temporal.py', 'possession/boundary.py', 'possession/dead_ball_rules.py',
        'scripts/research/video_ball/export_candidate.py', 'scripts/research/video_ball/fuse_detections.py',
        'starter/load_smart_tagging.py']
    source_hashes = {name:sha256(ROOT / name) for name in sources}
    frozen = {path:sha256(path) for path in model_files + [args.ball, args.ball.with_suffix('.json'),
        parent_dir / 'selected.pt', parent_dir / 'fixed_protocol.json']}
    from scripts.research.spatial_memory.run_external import torch_setup
    from scripts.run_local import selected_model, symmetric_probability, viterbi_labels
    from possession.features_v25 import build_features
    from possession.spatial_memory import build_features as memory_features
    from possession.inference_cache import cached_block
    from possession.dead_ball_rules import apply_oob
    from scripts.research.video_ball.export_candidate import automatic_prekickoff_dead, write_csv
    from scripts.research.video_ball.fuse_detections import read_exact_tracking
    from starter.load_smart_tagging import load as load_tagging
    import torch
    device = torch_setup(args.device)
    parent, parent_state = selected_model(parent_dir, device)
    expert, expert_state = selected_model(args.model, device)
    if expert_state['config']['sample_step'] != 2 or expert_state['decoder_costs'] != parent_state['decoder_costs']:
        raise ValueError('Expert clock or decoder differs from the external protocol')
    inputs = {k:sha256(paths[k]) for k in ('atd_raw', 'atd_meta', 'smart_tagging')}
    ball_hash = sha256(args.ball)
    receipt = json.loads(args.ball.with_suffix('.json').read_text())
    if (receipt['output_sha256'] != ball_hash or receipt['calibration_used_provided_match'] or
            receipt['original_tracking_sha256'] != inputs['atd_raw']):
        raise ValueError('Coordinate artifact differs from its verified provenance')
    players, original, teams = read_exact_tracking(args.data)
    ball = np.load(args.ball, allow_pickle=False)
    seen = np.isfinite(original).all(1)
    if ball.shape != original.shape or ball.dtype != np.float32 or not np.array_equal(ball[seen], original[seen]):
        raise ValueError('Original supplied ball observations changed')
    feature_files = ['possession/external.py', 'possession/data.py', 'scripts/run_local.py',
        'games.py', 'possession/inference_cache.py', 'possession/features_v08.py',
        'possession/features_v09.py', 'possession/features_v13.py', 'possession/features_v25.py']
    feature_sources = {name:source_hashes[name] for name in feature_files}
    raw_inputs = {k:inputs[k] for k in ('atd_meta', 'atd_raw')}
    raw_inputs['ball_hypotheses'] = ball_hash
    n, pitch = GAMES['BAR-ATH']['n_video_frames'], GAMES['BAR-ATH']['pitch']
    labels = np.full(n, 'X', dtype='<U1')
    for half, first, last in half_video_blocks('BAR-ATH'):
        key = dict(raw_inputs=raw_inputs, sources=feature_sources, game='BAR-ATH',
            first=first, last=last, pitch=list(pitch), sample_step=1, feature_version='v25',
            numpy=np.__version__, scipy=scipy.__version__)
        matrix, names, _ = cached_block(ROOT / 'data/inference-features', key,
            lambda:build_features(players[first:last + 1], teams, ball[first:last + 1], pitch, step=1), last - first + 1)
        if names != parent_state['names']:
            raise ValueError('Parent feature schema differs')
        p = symmetric_probability(parent, matrix, names, parent_state['mean'], parent_state['scale'], device)
        memory, memory_names = memory_features(players[first:last + 1], teams, ball[first:last + 1], pitch)
        expert_names = names[:664] + memory_names
        if expert_names != expert_state['names']:
            raise ValueError('Expert feature schema differs')
        x = np.concatenate([matrix[::2, :664], memory[::2]], axis=1)
        q = symmetric_probability(expert, x, expert_names, expert_state['mean'], expert_state['scale'], device)
        probability = mix_probabilities(p, q, memory[::2], memory_names, selection['selected']['strength'])
        labels[first:last + 1] = viterbi_labels(probability, *parent_state['decoder_costs'])
        print(f'Frozen observation-gate inference complete: half {half}', flush=True)
        del matrix, memory, x, p, q, probability
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / 'raw.csv', labels)
    after, kickoff = automatic_prekickoff_dead(labels, load_tagging(data_dir=str(args.data)), half_video_blocks('BAR-ATH'))
    graded = np.zeros(n, bool)
    for _, first, last in half_video_blocks('BAR-ATH'):
        graded[first:last + 1] = True
    final, oob = apply_oob(after, original, pitch, graded)
    write_csv(args.out / 'CoreOrigin.csv', final)
    if any(sha256(path) != digest for path, digest in frozen.items()):
        raise ValueError('Frozen model, selection or coordinate artifact changed during inference')
    if any(sha256(ROOT / name) != digest for name, digest in source_hashes.items()):
        raise ValueError('Executed source changed during inference')
    if inputs != {k:sha256(paths[k]) for k in inputs}:
        raise ValueError('Supplied inputs changed during inference')
    result = dict(rows=n, output_sha256=sha256(args.out / 'CoreOrigin.csv'), raw_sha256=sha256(args.out / 'raw.csv'),
        selection_sha256=sha256(args.model / 'selection.json'), selected_model_sha256=selection['selected_model_sha256'],
        parent_sha256=selection['parent_sha256'], input_sha256=inputs, ball_hypotheses_sha256=ball_hash,
        source_sha256=source_hashes, device=str(device), versions=dict(torch=torch.__version__, cuda=torch.version.cuda,
            numpy=np.__version__, scipy=scipy.__version__), source_frozen_throughout=True,
        training_data_and_truth_guard_verified=True, unexpected_forbidden_access=denied,
        training_on_provided_match=False, public_or_private_possession_truth_read=False,
        automatic_prekickoff_dead_intervals=kickoff, original_oob_frames_changed=int(oob.sum()),
        head_orientation_or_gaze_used=False, full_match_spell_f1=None, competition_target_verified=False)
    (args.out / 'manifest.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['rows', 'output_sha256', 'competition_target_verified']}), flush=True)


if __name__ == '__main__':
    main()
