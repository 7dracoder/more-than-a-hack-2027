"""Reproduce all frozen external D-calibration candidates and selection guards."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from possession.data import sha256
from scripts.research.spatial_memory.frozen_development import load, ORIGIN, FALLBACK
from scripts.research.spatial_memory.select_dead_calibration import scores, make_row, PROTOCOL
from scripts.research.spatial_memory.run_external import dump


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    args = parser.parse_args()
    origin,fallback,parts,hashes,denied = load(args.study)
    selection = json.loads((args.study / 'selection.json').read_text())
    assert selection['complete'] and selection['zero_offset_control_reproduced']
    assert not selection['provided_match_used'] and not selection['public_labels_used']
    assert not selection['unexpected_competition_access'] and not selection['model_weights_changed']
    plan = selection['plan']
    assert plan['inputs'] == hashes and plan['protocol'] == PROTOCOL
    for path,digest in {**plan['sources'], **plan['inputs']}.items():
        assert sha256(ROOT / path) == digest, path
    for key,name in [('model_sha256','selected.pt'),('model_protocol_sha256','fixed_protocol.json'),('preceding_selection_sha256','selection.json')]:
        assert sha256(ORIGIN / name) == plan[key]
    assert sha256(FALLBACK / 'selection.json') == plan['fallback_selection_sha256']
    import torch
    state = torch.load(ORIGIN / 'selected.pt',map_location='cpu',weights_only=False)
    reproduced = []
    for expected in selection['candidates']:
        strong,weak = scores(parts,state['decoder_costs'],expected['dead_offset'])
        result = make_row(strong,weak,fallback['selected'],expected['dead_offset'])
        assert result == expected, expected['dead_offset']
        reproduced.append(dict(offset=result['dead_offset'],objective=result['objective'],eligible=result['eligible']))
    best = max((r for r in selection['candidates'] if r['eligible']),key=lambda r:r['objective'])
    assert best == selection['selected']
    result = dict(complete=True,all_candidates_reproduced=True,all_guards_recomputed=True,
        source_and_input_hashes_verified=True,weights_unchanged=True,candidates=reproduced,
        selection_sha256=sha256(args.study / 'selection.json'),verifier_sha256=sha256(Path(__file__)),
        provided_match_used=False,public_labels_used=False,unexpected_competition_access=denied,
        independent_test=False,full_match_spell_f1=None,competition_target_verified=False)
    dump(args.study / 'external_verification.json',result)
    print(json.dumps(dict(complete=True,offset=best['dead_offset'],objective=best['objective'])),flush=True)


if __name__ == '__main__':
    main()
