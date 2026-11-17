"""Export an externally frozen winner without training-data or truth access.

Only supplied coordinate/tag inputs and a previously verified ball artifact are
read. The selection must be complete before this command is invoked.
"""
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


def inference_guard(out, files):
    data = (ROOT / 'data').resolve()
    truth = (ROOT / 'evaluation/gt').resolve()
    cache = (data / 'inference-features').resolve()
    out = out.resolve()
    files = {Path(p).resolve() for p in files}
    if not out.is_relative_to(data) or out.is_relative_to(data / 'external'):
        raise ValueError('Export must stay in a separate ignored data directory')
    denied = []

    def guard(event, values):
        if event == 'open' and values and isinstance(values[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(values[0])).resolve()
            if path.is_relative_to(truth) or (path.is_relative_to(data) and path not in files and
                    not path.is_relative_to(out) and not path.is_relative_to(cache)):
                denied.append(str(path))
                raise PermissionError('Frozen inference cannot open training data or possession truth')

    sys.addaudithook(guard)
    for path in (truth / '__guard_probe__.csv', data / 'external/__guard_probe__.csv'):
        try:
            open(path, 'rb')
            raise AssertionError('Inference isolation failed')
        except PermissionError:
            pass
    denied.clear()
    return denied


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--ball', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=ROOT / 'data')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    paths = asset_paths(data_dir=str(args.data))
    selection_path = args.study / 'memory_replacement/selection.json'
    preceding_path = args.study / 'final_external_selection.json'
    training_path = args.study / 'selection.json'
    preparation_path = args.study / 'training_plan.json'
    fixed_path = args.study / 'fixed_protocol.json'
    replacement_model = args.study / 'memory_replacement/selected.pt'
    files = [paths[k] for k in ('atd_raw', 'atd_meta', 'smart_tagging')]
    files += [args.ball, args.ball.with_suffix('.json'), selection_path, preceding_path,
        training_path, preparation_path, fixed_path, args.study / 'selected.pt',
        replacement_model, replacement_model.with_name('fixed_protocol.json')]
    denied = inference_guard(args.out, files)
    selection = json.loads(selection_path.read_text())
    preceding = json.loads(preceding_path.read_text())
    training = json.loads(training_path.read_text())
    prepared = json.loads(preparation_path.read_text())
    frozen_files = {p:sha256(p) for p in [selection_path, preceding_path, training_path, preparation_path, fixed_path,
        args.ball, args.ball.with_suffix('.json')]}
    if sha256(preceding_path) != selection['preceding_selection_sha256'] or sha256(training_path) != preceding['training_selection_sha256']:
        raise ValueError('External selection chain changed')
    for record in (selection, preceding, training):
        if (not record['complete'] or record['provided_match_used'] or record['public_labels_used'] or
                record['unexpected_competition_access']):
            raise ValueError('Need frozen external-only selection with verified isolation')
    for name, expected in prepared['sources'].items():
        if sha256(ROOT / name) != expected:
            raise ValueError('Training/preparation source changed')
    if sha256(fixed_path) != training['protocol_sha256']:
        raise ValueError('Training protocol changed')
    if sha256(ROOT / 'scripts/research/spatial_memory/compare_memory_replacement.py') != selection['plan']['source_sha256']:
        raise ValueError('Executed replacement recipe differs from its external protocol')
    if sha256(ROOT / 'scripts/research/spatial_memory/compare_natural_candidates.py') != preceding['source_sha256']:
        raise ValueError('Executed final comparison source changed')
    prior_path = ROOT / 'models/coreorigin-temporal-fusion-v1/selection.json'
    if (sha256(prior_path) != selection['plan']['prior_selection_sha256'] or
            sha256(prior_path) != preceding['plan']['incumbent_selection_sha256']):
        raise ValueError('Prior frozen recipe changed')
    frozen_files[prior_path] = sha256(prior_path)
    kind = selection['selected']['kind']
    if kind not in ('parent', 'incumbent_temporal_fusion', 'natural_adaptation',
            'memory_replacement_development_selected_checkpoint', 'memory_replacement_fixed_final_epoch_checkpoint'):
        raise ValueError('Unknown externally selected candidate')
    from scripts.research.spatial_memory.run_external import torch_setup
    from scripts.run_local import selected_model, symmetric_probability, viterbi_labels
    from possession.features_v25 import build_features
    from possession.spatial_memory import build_features as memory_features
    from possession.inference_cache import cached_block
    from possession.dead_ball_rules import apply_oob
    from scripts.research.video_ball.export_candidate import automatic_prekickoff_dead, write_csv
    from scripts.research.video_ball.fuse_detections import read_exact_tracking
    from starter.load_smart_tagging import load as load_tagging
    device = torch_setup(args.device)
    import torch
    parent_dir = ROOT / 'models/coreorigin-video-v1'
    if sha256(parent_dir / 'selected.pt') != prepared['comparison_sha256']:
        raise ValueError('Parent changed')
    parent, parent_state = selected_model(parent_dir, device)
    if parent_state['decoder_costs'] != selection['plan']['decoder_costs']:
        raise ValueError('Parent decoder differs from the externally selected recipe')
    frozen_files[parent_dir / 'selected.pt'] = prepared['comparison_sha256']
    frozen_files[parent_dir / 'fixed_protocol.json'] = sha256(parent_dir / 'fixed_protocol.json')
    expert, expert_state, strength = None, None, None
    if kind == 'incumbent_temporal_fusion':
        expert_dir = ROOT / 'models/coreorigin-robust-v13'
        expected = prepared['initialization_sha256']
    elif kind == 'natural_adaptation':
        expert_dir, expected = args.study, training['selected_sha256']
        strength = training['best']['strength']
    elif kind.startswith('memory_replacement_'):
        expert_dir, expected = replacement_model.parent, selection['selected_model_sha256']
    else:
        expert_dir, expected = None, None
    if expert_dir is not None:
        if sha256(expert_dir / 'selected.pt') != expected:
            raise ValueError('Selected expert changed')
        expert, expert_state = selected_model(expert_dir, device)
        frozen_files[expert_dir / 'selected.pt'] = expected
        frozen_files[expert_dir / 'fixed_protocol.json'] = sha256(expert_dir / 'fixed_protocol.json')
    inputs = {k:sha256(paths[k]) for k in ('atd_raw', 'atd_meta', 'smart_tagging')}
    ball_hash = sha256(args.ball)
    receipt = json.loads(args.ball.with_suffix('.json').read_text())
    if (receipt['output_sha256'] != ball_hash or receipt['calibration_used_provided_match'] or
            receipt['original_tracking_sha256'] != inputs['atd_raw']):
        raise ValueError('Ball artifact provenance differs')
    players, original, teams = read_exact_tracking(args.data)
    ball = np.load(args.ball, allow_pickle=False)
    observed = np.isfinite(original).all(1)
    if ball.shape != original.shape or ball.dtype != np.float32 or not np.array_equal(ball[observed], original[observed]):
        raise ValueError('Original ball observations changed')
    source_files = ['possession/external.py', 'possession/data.py', 'scripts/run_local.py',
        'games.py', 'possession/inference_cache.py', 'possession/features_v08.py',
        'possession/features_v09.py', 'possession/features_v13.py', 'possession/features_v25.py']
    feature_sources = {name:sha256(ROOT / name) for name in source_files}
    raw_inputs = {k:inputs[k] for k in ('atd_meta', 'atd_raw')}
    raw_inputs['ball_hypotheses'] = ball_hash
    n, pitch = GAMES['BAR-ATH']['n_video_frames'], GAMES['BAR-ATH']['pitch']
    labels = np.full(n, 'X', dtype='<U1')
    for half, first, last in half_video_blocks('BAR-ATH'):
        key = dict(raw_inputs=raw_inputs, sources=feature_sources, game='BAR-ATH',
            first=first, last=last, pitch=list(pitch), sample_step=1,
            feature_version='v25', numpy=np.__version__, scipy=scipy.__version__)
        matrix, names, _ = cached_block(ROOT / 'data/inference-features', key,
            lambda:build_features(players[first:last + 1], teams, ball[first:last + 1], pitch, step=1), last - first + 1)
        if names != parent_state['names']:
            raise ValueError('Parent input schema differs')
        p = symmetric_probability(parent, matrix, names, parent_state['mean'], parent_state['scale'], device)
        if expert is not None:
            memory, memory_names = memory_features(players[first:last + 1], teams, ball[first:last + 1], pitch)
            if kind == 'incumbent_temporal_fusion':
                from scripts.research.spatial_memory.select_temporal_fusion import fusion_probability
                if names[:664] != expert_state['names']:
                    raise ValueError('Incumbent expert prefix differs')
                q = symmetric_probability(expert, matrix[::2, :664], expert_state['names'], expert_state['mean'], expert_state['scale'], device)
                q = np.repeat(q, 2, axis=0)[:len(p)]
                p = fusion_probability(p, q, memory, memory_names, 'long_missing', .75)
            else:
                expert_names = names[:664] + memory_names
                if expert_names != expert_state['names']:
                    raise ValueError('Adapted expert schema differs')
                x = np.concatenate([matrix[::2, :664], memory[::2]], axis=1)
                q = symmetric_probability(expert, x, expert_names, expert_state['mean'], expert_state['scale'], device)
                if kind == 'natural_adaptation':
                    p = (1. - strength) * p + strength * np.repeat(q, 2, axis=0)[:len(p)]
                else:
                    from scripts.research.spatial_memory.compare_memory_replacement import replacement_probability
                    p = replacement_probability(p, q, memory[::2], memory_names)
                del x
            del memory, q
        labels[first:last + 1] = viterbi_labels(p, *parent_state['decoder_costs'])
        print(f'Frozen {kind}: half {half} inference complete', flush=True)
        del matrix, p
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / 'raw.csv', labels)
    after, kickoff = automatic_prekickoff_dead(labels, load_tagging(data_dir=str(args.data)), half_video_blocks('BAR-ATH'))
    graded = np.zeros(n, bool)
    for _, first, last in half_video_blocks('BAR-ATH'):
        graded[first:last + 1] = True
    final, oob = apply_oob(after, original, pitch, graded)
    write_csv(args.out / 'CoreOrigin.csv', final)
    source_files += ['possession/spatial_memory.py', 'possession/temporal.py', 'possession/boundary.py',
        'possession/dead_ball_rules.py', 'scripts/research/video_ball/export_candidate.py',
        'scripts/research/video_ball/fuse_detections.py', 'scripts/research/spatial_memory/select_temporal_fusion.py',
        'scripts/research/spatial_memory/compare_memory_replacement.py', 'starter/load_smart_tagging.py']
    dump = dict(rows=n, kind=kind, output_sha256=sha256(args.out / 'CoreOrigin.csv'),
        raw_sha256=sha256(args.out / 'raw.csv'), final_selection_sha256=sha256(selection_path),
        parent_sha256=prepared['comparison_sha256'], expert_sha256=expected,
        input_sha256=inputs, ball_hypotheses_sha256=ball_hash,
        source_sha256={name:sha256(ROOT / name) for name in source_files}, exporter_sha256=sha256(Path(__file__)),
        versions=dict(numpy=np.__version__, scipy=scipy.__version__), device=str(device),
        automatic_prekickoff_dead_intervals=kickoff, original_oob_frames_changed=int(oob.sum()),
        training_data_and_truth_guard_verified=True, unexpected_forbidden_access=denied,
        training_on_provided_match=False, public_or_private_possession_truth_read=False,
        head_orientation_or_gaze_used=False, full_match_spell_f1=None, competition_target_verified=False)
    if inputs != {k:sha256(paths[k]) for k in inputs}:
        raise ValueError('Supplied inputs changed during inference')
    if any(sha256(path) != expected_hash for path, expected_hash in frozen_files.items()):
        raise ValueError('Frozen selection, models or ball artifact changed during inference')
    dump['versions'].update(torch=torch.__version__, cuda=torch.version.cuda)
    (args.out / 'manifest.json').write_text(json.dumps(dump, indent=2), encoding='utf-8')
    print(json.dumps({k:dump[k] for k in ['rows', 'kind', 'output_sha256', 'competition_target_verified']}), flush=True)


if __name__ == '__main__':
    main()
