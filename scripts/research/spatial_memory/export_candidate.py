"""Apply one externally frozen spatial-memory study result to BAR-ATH.

No public/private possession labels are opened, and no parameters are selected
here. Original externally selected possession rules remain unchanged.
"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import scipy

from games import GAMES, asset_paths, half_video_blocks
from possession.data import sha256
from possession.dead_ball_rules import apply_oob
from possession.features_v25 import build_features as parent_features
from possession.inference_cache import cached_block
from possession.spatial_memory import build_features as memory_features
from scripts.research.spatial_memory.run_external import blend, corrected_probability
from scripts.research.video_ball.export_candidate import automatic_prekickoff_dead, write_csv
from scripts.research.video_ball.fuse_detections import read_exact_tracking
from starter.load_smart_tagging import load as load_tagging


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--ball', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=ROOT / 'data')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    if not args.out.resolve().is_relative_to((ROOT / 'data').resolve()):
        raise ValueError('Predictions must remain in ignored data/')
    import torch
    import xgboost as xgb
    from scripts.run_local import selected_model, symmetric_probability, viterbi_labels
    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    selection_path = args.study / 'selection.json'
    selection_hash = sha256(selection_path)
    selection = json.loads(selection_path.read_text(encoding='utf-8'))
    if (not selection['complete'] or selection['competition_data_read'] or
            selection['public_labels_read'] or selection['unexpected_competition_access']):
        raise ValueError('Selection must be external only, with a verified isolation guard')
    for source, expected in selection['source_sha256'].items():
        if sha256(ROOT / source) != expected:
            raise ValueError('Executed external-study source changed')
    parent = ROOT / 'models/coreorigin-video-v1'
    if sha256(parent / 'selected.pt') != selection['parent_sha256']:
        raise ValueError('Frozen temporal parent differs')
    chosen = selection['selected']
    family = selection.get('correction_family')
    booster_path = (ROOT / selection['fallback_dir'] / 'selected.pt' if family == 'temporal_fusion'
                    else args.study / (chosen['kind'] + '.ubj'))
    if sha256(booster_path) != selection['selected_model_sha256']:
        raise ValueError('Externally selected correction model changed')
    booster, fallback, fallback_state = None, None, None
    if family == 'temporal_fusion':
        fallback, fallback_state = selected_model(ROOT / selection['fallback_dir'], device)
        if (fallback_state['names'] != selection['fallback_names'] or
                fallback_state['config']['sample_step'] != selection['fallback_sample_step']):
            raise ValueError('Frozen fallback feature/timing schema differs')
    else:
        booster = xgb.XGBClassifier()
        booster.load_model(booster_path)
        booster.set_params(device='cpu', n_jobs=4)
    model, state = selected_model(parent, device)
    paths = asset_paths(data_dir=str(args.data))
    inputs = {name:sha256(paths[name]) for name in ('atd_raw', 'atd_meta', 'smart_tagging')}
    receipt = json.loads(args.ball.with_suffix('.json').read_text(encoding='utf-8'))
    ball_hash = sha256(args.ball)
    if (receipt['output_sha256'] != ball_hash or receipt['calibration_used_provided_match'] or
            receipt['original_tracking_sha256'] != inputs['atd_raw']):
        raise ValueError('Unverified dual-feed ball artifact')
    players, original, teams = read_exact_tracking(args.data)
    ball = np.load(args.ball, allow_pickle=False)
    seen = np.isfinite(original).all(1)
    if ball.shape != original.shape or ball.dtype != np.float32 or not np.array_equal(ball[seen], original[seen]):
        raise ValueError('Original tactical ball observations must be preserved')
    n, pitch = GAMES['BAR-ATH']['n_video_frames'], GAMES['BAR-ATH']['pitch']
    labels = np.full(n, 'X', dtype='<U1')
    source_files = ['possession/external.py', 'possession/data.py', 'scripts/run_local.py',
                    'games.py', 'possession/inference_cache.py', 'possession/features_v08.py',
                    'possession/features_v09.py', 'possession/features_v13.py', 'possession/features_v25.py']
    feature_sources = {name:sha256(ROOT / name) for name in source_files}
    raw_inputs = {name:inputs[name] for name in ('atd_meta', 'atd_raw')}
    raw_inputs['ball_hypotheses'] = ball_hash
    for half, first, last in half_video_blocks('BAR-ATH'):
        key = dict(raw_inputs=raw_inputs, sources=feature_sources, game='BAR-ATH',
                   first=first, last=last, pitch=list(pitch), sample_step=1,
                   feature_version='v25', numpy=np.__version__, scipy=scipy.__version__)
        matrix, names, _ = cached_block(ROOT / 'data/inference-features', key,
            lambda:parent_features(players[first:last + 1], teams, ball[first:last + 1], pitch, step=1),
            last - first + 1)
        if names != state['names']:
            raise ValueError('Parent features differ')
        p = symmetric_probability(model, matrix, names, state['mean'], state['scale'], device)
        memory, names = memory_features(players[first:last + 1], teams, ball[first:last + 1], pitch)
        if names != selection['memory_names']:
            raise ValueError('Memory schema differs from external training')
        if family == 'temporal_fusion':
            from scripts.research.spatial_memory.select_temporal_fusion import fusion_probability
            k, step = len(fallback_state['names']), selection['fallback_sample_step']
            if state['names'][:k] != fallback_state['names']:
                raise ValueError('Fallback does not match a verified prefix of the parent features')
            q = symmetric_probability(fallback, matrix[::step, :k], fallback_state['names'],
                                      fallback_state['mean'], fallback_state['scale'], device)
            q = np.repeat(q, step, axis=0)[:len(p)]
            final_probability = fusion_probability(p, q, memory, names, chosen['mode'], chosen['strength'])
        elif family == 'full_context':
            from scripts.research.spatial_memory.fit_full_context import full_probability
            if selection['base_names'] != state['names']:
                raise ValueError('Full-context correction expects a different kinematic bank')
            corrected = full_probability(booster, matrix, state['names'], memory, names, selection['use_memory'])
            final_probability = blend(p, corrected, chosen['strength'])
        elif family == 'independent':
            from scripts.research.spatial_memory.fit_independent import independent_probability
            corrected = independent_probability(booster, memory, names, selection['selected_columns'])
            final_probability = blend(p, corrected, chosen['strength'])
        else:
            corrected = corrected_probability(booster, p, memory, names, chosen['kind'])
            final_probability = blend(p, corrected, chosen['strength'])
        labels[first:last + 1] = viterbi_labels(final_probability, *selection['decoder_costs'])
        print(f'Externally frozen {chosen["kind"]} inference complete for half {half}', flush=True)
        del matrix, p, memory, final_probability
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / 'raw.csv', labels)
    after, intervals = automatic_prekickoff_dead(labels, load_tagging(data_dir=str(args.data)), half_video_blocks('BAR-ATH'))
    graded = np.zeros(n, bool)
    for _, first, last in half_video_blocks('BAR-ATH'):
        graded[first:last + 1] = True
    final, oob = apply_oob(after, original, pitch, graded)
    write_csv(args.out / 'CoreOrigin.csv', final)
    if sha256(selection_path) != selection_hash or inputs != {name:sha256(paths[name]) for name in inputs}:
        raise ValueError('Frozen selection or supplied inputs changed')
    result = dict(rows=n, output_sha256=sha256(args.out / 'CoreOrigin.csv'),
                  raw_sha256=sha256(args.out / 'raw.csv'), selection_sha256=selection_hash,
                  parent_sha256=selection['parent_sha256'], correction_model_sha256=selection['selected_model_sha256'],
                  external_selected_kind=chosen['kind'], external_selected_strength=chosen['strength'],
                  external_development_scores=chosen['scores'],
                  ball_hypotheses_sha256=ball_hash, input_sha256=inputs,
                  feature_sources=feature_sources, source_sha256=sha256(__file__),
                  memory_source_sha256=sha256(ROOT / 'possession/spatial_memory.py'),
                  automatic_prekickoff_dead_intervals=intervals,
                  original_observation_oob_frames_changed=int(oob.sum()),
                  device=str(device), versions=dict(torch=torch.__version__, xgboost=xgb.__version__, numpy=np.__version__),
                  training_on_provided_match=False, public_or_private_ground_truth_read=False,
                  head_orientation_or_gaze_used=False, video_prediction_comparison_performed=False,
                  full_match_spell_f1=None, target_verified=False,
                  status='Externally selected experiment; no BAR-ATH improvement claim')
    (args.out / 'manifest.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({key:result[key] for key in ['rows', 'output_sha256', 'external_selected_kind',
                    'external_selected_strength', 'full_match_spell_f1']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
