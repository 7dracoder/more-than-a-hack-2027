"""Paired external possession audit of receiver-context adaptation against v4."""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from possession.data import sha256
from possession.observation_gate import mix_probabilities,observation_loss
from possession.uncertain_decode import decode_labels
from possession.external_spell_uncertainty import spell_information,paired_draws
from scripts.research.spatial_memory.external_guard import restrict_external
from scripts.research.spatial_memory.run_external import dump,score_parts,torch_setup
from scripts.research.spatial_memory.train_real_broadcast_memory import STRONG,TRANSFER,VIEWS

BASE = ROOT / 'data/research/continuity-calibration-20261114'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--audit-name',default='paired_uncertainty')
    parser.add_argument('--device',default='cuda')
    args = parser.parse_args()
    args.run = args.run.resolve()
    if Path(args.audit_name).name != args.audit_name or not args.audit_name:
        raise ValueError('Audit name must be a single directory name')
    selected = json.loads((args.run / 'selection.json').read_text())
    verified = json.loads((args.run / 'external_verification.json').read_text())
    assert selected['complete'] and verified['complete'] and verified['selected_scores_reproduced']
    assert verified['selection_sha256'] == sha256(args.run / 'selection.json')
    assert not selected['provided_match_used'] and not selected['public_labels_used']
    assert not selected['unexpected_competition_access']
    split = selected['plan']['split']
    fallback = json.loads((ROOT / 'models/coreorigin-continuity-v4/selection.json').read_text())
    specs = []
    for match in split['strong_development_matches']:
        for half in (0,1):
            for view in VIEWS:
                stem = STRONG / 'cache' / f'{match}_{half}_{view}'
                specs.append(dict(domain='strong',view=view,x=stem.with_suffix('.npy'),
                    truth=stem.with_name(stem.name+'_truth.npy'),parent=stem.with_name(stem.name+'_parent.npy')))
    for match in split['weak_development_matches']:
        for half in (1,2):
            stem = f'{match}_{half}_detected'
            specs.append(dict(domain='weak',x=TRANSFER / 'features' / (stem+'.npy'),
                truth=TRANSFER / 'features' / (stem+'_weak.npz'),parent=TRANSFER / 'predictions' / (stem+'.npz')))
    allowed = [BASE / 'probability_freeze.json']
    counts = {'strong':0,'weak':0}
    for h in specs:
        domain = h['domain']
        h['baseline'] = BASE / 'probabilities' / f'{domain}_{counts[domain]}.npz'
        counts[domain] += 1
        allowed += [h[k] for k in ('x','truth','parent','baseline')] + [h['x'].with_suffix('.json')]
    for h in specs:
        candidates=[ROOT/p for p in selected['plan']['caches'] if Path(p).name==h['x'].name and 'receiver-context-20261114' in Path(p).parts]
        assert len(candidates)==1
        h['context']=candidates[0]
        allowed += [h['context'],h['context'].with_suffix('.json')]
    denied = restrict_external(args.run,allowed)
    out = args.run / args.audit_name
    out.mkdir(exist_ok=True)
    if (out / 'results.json').exists():
        raise ValueError('Preserve completed paired audit')
    freeze = json.loads((BASE / 'probability_freeze.json').read_text())
    sources = ['scripts/research/spatial_memory/audit_receiver_context_uncertainty.py',
        'possession/external_spell_uncertainty.py','possession/observation_gate.py','possession/uncertain_decode.py',
        'scripts/research/spatial_memory/run_external.py','scripts/research/spatial_memory/external_guard.py',
        'evaluation/possession_metrics.py']
    plan = dict(n_draws=1000,seed=0,percentiles=[5,95],source_sha256={p:sha256(ROOT / p) for p in sources},
        model_sha256=sha256(args.run / 'selected.pt'),selection_sha256=sha256(args.run / 'selection.json'),
        verification_sha256=sha256(args.run / 'external_verification.json'),
        baseline_selection_sha256=sha256(ROOT / 'models/coreorigin-continuity-v4/selection.json'),
        baseline_probability_freeze_sha256=sha256(BASE / 'probability_freeze.json'),
        resampling_unit='Whole GT possession with identical sampled indices across both models and all three correlated views.',
        decision='Within-fixture support requires positive paired 5th percentile and passing all selection guards; not proof of independent or BAR-ATH improvement.',
        limitation='Post-selection, conditional on selected epoch, no adjustment for epoch search, one previously opened strong fixture; no match-level generalization claim.',
        provided_match_used=False,public_labels_used=False,independent_test=False)
    assert plan['model_sha256'] == selected['selected_model_sha256']
    if (out / 'plan.json').exists():
        assert json.loads((out / 'plan.json').read_text()) == plan
    dump(out / 'plan.json',plan)
    from scripts.run_local import selected_model
    from possession.temporal import symmetric_probability
    model,state = selected_model(args.run,torch_setup(args.device))
    files = {}
    for i,h in enumerate(specs):
        for key in ('x','truth','parent'):
            path = h[key]
            assert sha256(path) == selected['plan']['caches'][str(path.relative_to(ROOT))]
            files[str(path.relative_to(ROOT))] = sha256(path)
        assert sha256(h['baseline']) == freeze['files'][h['baseline'].name]
        files[str(h['baseline'].relative_to(ROOT))] = sha256(h['baseline'])
        metadata = json.loads(h['x'].with_suffix('.json').read_text())
        context_meta=json.loads(h['context'].with_suffix('.json').read_text())
        for path in [h['context'],h['context'].with_suffix('.json')]:
            assert sha256(path)==selected['plan']['caches'][str(path.relative_to(ROOT))]
            files[str(path.relative_to(ROOT))]=sha256(path)
        assert metadata['names'][:664]+metadata['names'][960:]+context_meta['features']==state['names']
        raw = np.load(h['x'],mmap_mode='r')
        x = raw if h['domain'] == 'strong' else raw[::2]
        if h['domain'] == 'strong':
            truth = np.load(h['truth'])
            parent = np.load(h['parent'])
        else:
            with np.load(h['truth']) as data:
                h['ticks'],truth = data['ticks'],data['truth']
            with np.load(h['parent']) as data:
                parent = data['parent']
        memory=x[:,960:]
        x=np.concatenate([x[:,:664],memory,np.load(h['context'])],axis=1)
        q = symmetric_probability(model,x,state['names'],state['mean'],state['scale'],next(model.parameters()).device)
        p = mix_probabilities(parent,q,memory,metadata['names'][960:],1.)
        gate = np.repeat(observation_loss(memory,metadata['names'][960:]),2)[:len(p)]
        with np.load(h['baseline']) as base:
            np.testing.assert_array_equal(gate,base['gate'])
            h['baseline_labels'] = decode_labels(base['probability'],gate,8.,*state['decoder_costs'])
        h['candidate_labels'] = decode_labels(p,gate,8.,*state['decoder_costs'])
        h['target'] = truth
        path = out / f'predictions_{i}.npz'
        np.savez_compressed(path,probability=p,gate=gate,labels=h['candidate_labels'])
        files[str(path.relative_to(ROOT))] = sha256(path)
        print(f'Frozen paired geometry audit: {h["domain"]} {i+1}',flush=True)
    flags,predicted_counts,reference = [],[],None
    points = {}
    for view in VIEWS:
        current = [h for h in specs if h['domain'] == 'strong' and h['view'] == view]
        rows,counts = [],[]
        points[view] = {}
        for key in ('baseline_labels','candidate_labels'):
            truth,predicted = [h['target'] for h in current],[h[key] for h in current]
            f,n,gt = spell_information(truth,predicted)
            if reference is None:
                reference = gt
            assert gt == reference
            result = score_parts(truth,predicted)
            expected = fallback['selected'] if key == 'baseline_labels' else selected['selected']
            assert abs(result['spell_f1_mean']-expected['scores'][view]['spell_f1_mean']) < 1e-10
            points[view][key] = result
            rows.append(f);counts.append(n)
        flags.append(rows);predicted_counts.append(counts)
    draws,delta = paired_draws(np.asarray(flags),np.asarray(predicted_counts),1000,0)
    np.savez_compressed(out / 'draws.npz',view_scores=draws,mean_difference=delta)
    interval = np.percentile(delta,[5,95]).tolist()
    result = dict(complete=True,gt_possessions=len(reference),points=points,
        observed_mean_difference=selected['selected']['objective']-fallback['selected']['objective'],
        paired_difference_interval=interval,positive_difference_fraction=float((delta>0).mean()),
        per_view_difference_intervals={view:np.percentile(draws[i,1]-draws[i,0],[5,95]).tolist() for i,view in enumerate(VIEWS)},
        within_fixture_gain_supported=bool(interval[0]>0 and selected['selected']['eligible']),
        files=files,draws_sha256=sha256(out / 'draws.npz'),plan_sha256=sha256(out / 'plan.json'),
        provided_match_used=False,public_labels_used=False,unexpected_competition_access=denied,
        independent_test=False,full_match_spell_f1=None,competition_target_verified=False)
    for path,digest in plan['source_sha256'].items():
        assert sha256(ROOT / path) == digest,path
    for path,digest in files.items():
        assert sha256(ROOT / path) == digest,path
    dump(out / 'results.json',result)
    print(json.dumps({k:result[k] for k in ('observed_mean_difference','paired_difference_interval','within_fixture_gain_supported')}),flush=True)


if __name__ == '__main__':
    main()
