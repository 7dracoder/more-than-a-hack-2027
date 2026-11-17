"""Post-selection paired spell uncertainty audit; no BAR-ATH data are read."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from possession.data import sha256
from possession.observation_state_calibration import calibrate_dead_odds
from possession.uncertain_decode import decode_labels
from possession.external_spell_uncertainty import spell_information,paired_draws
from scripts.research.spatial_memory.frozen_development import load,ORIGIN
from scripts.research.spatial_memory.run_external import dump
from scripts.research.spatial_memory.train_real_broadcast_memory import VIEWS
from evaluation import possession_metrics as pm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True)
    args = parser.parse_args()
    origin,fallback,parts,hashes,denied = load(args.study)
    selection = json.loads((args.study / 'selection.json').read_text())
    verification = json.loads((args.study / 'external_verification.json').read_text())
    assert verification['complete'] and verification['all_candidates_reproduced']
    assert verification['selection_sha256'] == sha256(args.study / 'selection.json')
    out = args.study / 'paired_uncertainty'
    out.mkdir(exist_ok=True)
    if (out / 'results.json').exists():
        raise ValueError('Preserve completed uncertainty audit')
    sources = ['scripts/research/spatial_memory/audit_calibration_uncertainty.py',
        'possession/external_spell_uncertainty.py','possession/observation_state_calibration.py',
        'possession/uncertain_decode.py','scripts/research/spatial_memory/frozen_development.py',
        'evaluation/possession_metrics.py']
    plan = dict(n_draws=1000,seed=0,percentiles=[5,95],
        resampling_unit='Whole GT possession; same sampled indices shared across both models and all three correlated views.',
        scoring='Official greedy matched-GT flags per IoU and fixed prediction count with nonnegative FP clamp.',
        audit_stage='Post-selection reliability check, declared after the exploratory grid; not an independent or preregistered selection test.',
        minimum_material_gain=pm.SPELL_NOISE_BAND,
        promotion_rule='Require mean gain at least the official descriptive 0.04 spell noise band and positive paired 5th percentile; this is a conservative handoff rule, not a generalization guarantee.',
        limitation='One previously opened development fixture, conditional on the selected offset. No adjustment for searching the grid. Possession resampling cannot establish match-to-match generalization.',
        sources={p:sha256(ROOT / p) for p in sources},inputs=hashes,
        selection_sha256=sha256(args.study / 'selection.json'),
        selected_offset=selection['selected']['dead_offset'],
        provided_match_used=False,public_labels_used=False,independent_test=False)
    dump(out / 'plan.json',plan)
    import torch
    state = torch.load(ORIGIN / 'selected.pt',map_location='cpu',weights_only=False)
    flags,counts,reference = [],[],None
    for view in VIEWS:
        current = [h for h in parts if h['domain'] == 'strong' and h['view'] == view]
        model_flags,model_counts = [],[]
        for offset in [0.,selection['selected']['dead_offset']]:
            predicted = [decode_labels(calibrate_dead_odds(h['probability'],h['gate'],offset),
                h['gate'],8.,*state['decoder_costs']) for h in current]
            f,n,gt_spells = spell_information([h['truth'] for h in current],predicted)
            if reference is None:
                reference = gt_spells
            assert gt_spells == reference
            expected = fallback['selected'] if offset == 0 else selection['selected']
            point = float((2*f.sum(1)/(len(gt_spells)+n)).mean())
            assert abs(point-expected['scores'][view]['spell_f1_mean']) < 1e-12
            model_flags.append(f);model_counts.append(n)
        flags.append(model_flags);counts.append(model_counts)
    flags,counts = np.asarray(flags),np.asarray(counts)
    draws,delta = paired_draws(flags,counts,plan['n_draws'],plan['seed'])
    np.savez_compressed(out / 'draws.npz',view_scores=draws,mean_difference=delta,
        matched_gt_flags=flags,predicted_counts=counts)
    per_view = {}
    for i,view in enumerate(VIEWS):
        per_view[view] = dict(fallback_interval=np.percentile(draws[i,0],[5,95]).tolist(),
            candidate_interval=np.percentile(draws[i,1],[5,95]).tolist(),
            paired_difference_interval=np.percentile(draws[i,1]-draws[i,0],[5,95]).tolist())
    difference = selection['selected']['objective']-fallback['selected']['objective']
    interval = np.percentile(delta,[5,95]).tolist()
    promote = bool(difference >= plan['minimum_material_gain'] and interval[0] > 0)
    for path,digest in {**plan['sources'],**plan['inputs']}.items():
        assert sha256(ROOT / path) == digest,path
    result = dict(complete=True,gt_possessions=len(reference),per_view=per_view,
        observed_mean_difference=difference,paired_difference_interval=interval,
        mean_difference_positive_fraction=float((delta>0).mean()),
        draws_sha256=sha256(out / 'draws.npz'),plan_sha256=sha256(out / 'plan.json'),
        promote_candidate=promote,keep_v4_handoff=not promote,
        interpretation='Exploratory within-fixture uncertainty only; not BAR-ATH performance, independent confirmation or selection-adjusted inference.',
        provided_match_used=False,public_labels_used=False,unexpected_competition_access=denied,
        full_match_spell_f1=None,competition_target_verified=False)
    dump(out / 'results.json',result)
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
