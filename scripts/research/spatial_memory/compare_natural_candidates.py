"""Compare frozen external candidates on the new external visibility views.

No weights, thresholds, decoder settings or mixture strengths are tuned here.
The incumbent is evaluated with its exact full-rate memory gate, not a
downsampled approximation. J03WPY remains exploratory development data.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from possession.data import sha256
from scripts.research.spatial_memory.run_external import DEV, dump, isolate, score_parts, torch_setup
from scripts.research.spatial_memory.train_natural_visibility import PLAN as TRAINING_PLAN, seed_for, model_matrix


def freeze(out):
    path = out / 'comparison_plan.json'
    incumbent = ROOT / 'models/coreorigin-temporal-fusion-v1/selection.json'
    value = dict(candidates=['incumbent_temporal_fusion', 'parent', 'natural_adaptation'],
        incumbent_selection_sha256=sha256(incumbent),
        training_plan_sha256=sha256(out / 'training_plan.json'),
        source_sha256=sha256(Path(__file__)), views=TRAINING_PLAN['views'], match=DEV,
        decision='Highest mean external spell F1, subject to parent native score minus 0.01; ties retain incumbent.',
        weights_or_parameters_tuned=False, provided_match_used=False, public_labels_used=False,
        limitation='Same previously used external development fixture; not independent confirmation or BAR-ATH performance.')
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('Comparison protocol changed')
    else:
        dump(path, value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--freeze-only', action='store_true')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    denied = isolate(args.study)
    plan = freeze(args.study)
    if args.freeze_only:
        print('External candidate-comparison protocol frozen before training results.', flush=True)
        return
    result_path = args.study / 'final_external_selection.json'
    if result_path.exists():
        raise ValueError('The final comparison is already frozen')
    selection = json.loads((args.study / 'selection.json').read_text())
    if (not selection['complete'] or selection['provided_match_used'] or
            selection['public_labels_used'] or selection['unexpected_competition_access']):
        raise ValueError('Need completed isolated external selection')
    if sha256(args.study / 'fixed_protocol.json') != selection['protocol_sha256']:
        raise ValueError('Training protocol changed')
    initial = json.loads((args.study / 'training_plan.json').read_text())
    for name, expected in initial['sources'].items():
        if sha256(ROOT / name) != expected:
            raise ValueError('Training/preparation source changed')
    old = json.loads((ROOT / 'models/coreorigin-temporal-fusion-v1/selection.json').read_text())
    if old['parent_sha256'] != initial['comparison_sha256'] or old['selected_model_sha256'] != initial['initialization_sha256']:
        raise ValueError('Incumbent members differ from the independent training comparison')
    device = torch_setup(args.device)
    from scripts.run_local import selected_model, symmetric_probability, viterbi_labels
    from scripts.research.spatial_memory.select_temporal_fusion import fusion_probability
    from possession.external import read_sportec
    from possession.external_visibility import degrade
    from possession.spatial_memory import build_features as memory_features
    fallback, fallback_state = selected_model(ROOT / 'models/coreorigin-robust-v13', device)
    adapted, adapted_state = None, None
    if selection['best']['strength'] > 0:
        if sha256(args.study / 'selected.pt') != selection['selected_sha256']:
            raise ValueError('Selected adaptation changed')
        adapted, adapted_state = selected_model(args.study, device)
    profile_path = args.study / 'external_visibility.json'
    if sha256(profile_path) != initial['profile_sha256']:
        raise ValueError('External calibration changed')
    profile = json.loads(profile_path.read_text())
    folder = ROOT / 'data/external/sportec'
    paths = [next(folder.glob('*' + kind + '*' + DEV + '.xml')) for kind in ('matchinformation', 'positions')]
    pins = {x['name']:x['sha256'] for x in json.loads((folder / 'download_manifest.json').read_text())['files']}
    if any(sha256(p) != pins[p.name] for p in paths):
        raise ValueError('External raw data changed')
    truths = {view:[] for view in plan['views']}
    predictions = {kind:{view:[] for view in plan['views']} for kind in plan['candidates']}
    artifacts = {}
    for i, half in enumerate(read_sportec(*paths)):
        for view in plan['views']:
            path = args.study / 'cache' / f'{DEV}_{i}_{view}.npy'
            meta = json.loads(path.with_suffix('.json').read_text())
            if sha256(path) != meta['sha256']:
                raise ValueError('External cache changed')
            for name, expected in meta['auxiliary_sha256'].items():
                if sha256(path.parent / name) != expected:
                    raise ValueError('External comparison/target cache changed')
            x = np.load(path, mmap_mode='r', allow_pickle=False)
            truth = np.load(path.with_name(path.stem + '_truth.npy'))
            np.testing.assert_array_equal(truth, half.labels)
            parent = np.load(path.with_name(path.stem + '_parent.npy'))
            p, b = half.players, half.ball
            if view != 'native':
                p, b = degrade(p, half.teams, b, half.pitch, profile, seed_for(DEV, i, view),
                    mode='none' if view == 'masked' else 'mixed')
            memory, memory_names = memory_features(p, half.teams, b, half.pitch)
            np.testing.assert_array_equal(memory[::2], x[:, 960:])
            if memory_names != old['memory_names'] or meta['names'][:664] != fallback_state['names']:
                raise ValueError('Incumbent schema differs')
            q = symmetric_probability(fallback, x[:, :664], fallback_state['names'],
                fallback_state['mean'], fallback_state['scale'], device)
            q = np.repeat(q, 2, axis=0)[:len(truth)]
            incumbent = fusion_probability(parent, q, memory, memory_names,
                old['selected']['mode'], old['selected']['strength'])
            natural = parent.copy()
            if adapted is not None:
                r = symmetric_probability(adapted, model_matrix(x), adapted_state['names'],
                    adapted_state['mean'], adapted_state['scale'], device)
                r = np.repeat(r, 2, axis=0)[:len(truth)]
                strength = selection['best']['strength']
                natural = (1. - strength) * parent + strength * r
            for kind, prob in [('parent', parent), ('incumbent_temporal_fusion', incumbent), ('natural_adaptation', natural)]:
                predictions[kind][view].append(viterbi_labels(prob, *old['decoder_costs']))
            truths[view].append(truth)
            artifacts[path.name] = meta['sha256']
            print(f'Frozen external comparison: half {i + 1}, {view}', flush=True)
            del memory, q, incumbent, natural, x
    scores = {kind:{view:score_parts(truths[view], predictions[kind][view]) for view in plan['views']} for kind in plan['candidates']}
    for view in plan['views']:
        expected = selection['best']['scores'][view]['spell_f1_mean']
        if abs(scores['natural_adaptation'][view]['spell_f1_mean'] - expected) > 1e-10:
            raise ValueError('Selected adaptation development score failed exact reproduction')
    rows = [dict(kind=kind, scores=scores[kind],
        objective=float(np.mean([scores[kind][v]['spell_f1_mean'] for v in plan['views']])),
        eligible=scores[kind]['native']['spell_f1_mean'] >= scores['parent']['native']['spell_f1_mean'] - .01)
        for kind in plan['candidates']]
    chosen = max((r for r in rows if r['eligible']), key=lambda r:r['objective'])
    dump(result_path, dict(complete=True, selected=chosen, candidates=rows, plan=plan,
        training_selection_sha256=sha256(args.study / 'selection.json'),
        candidate_model_sha256=selection['selected_sha256'], artifacts=artifacts,
        source_sha256=sha256(Path(__file__)), unexpected_competition_access=denied,
        selected_score_reproduced=True, provided_match_used=False, public_labels_used=False,
        competition_target_achieved=False))
    print(json.dumps(dict(selected=chosen['kind'], objective=chosen['objective'])), flush=True)


if __name__ == '__main__':
    main()
