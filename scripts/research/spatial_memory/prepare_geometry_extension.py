"""Cache only missing external trajectory columns; reuse verified old inputs."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from possession.data import sha256
from scripts.research.spatial_memory.external_guard import restrict_external
from scripts.research.spatial_memory.run_external import dump

OLD = ROOT / 'data/research/real-broadcast-adaptation-20261114'
REFERENCE = ROOT / 'data/research/natural-visibility-20261114/cache/J03WMX_0_native.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    config_path = ROOT / 'configs/full_geometry_memory_adaptation.json'
    config = json.loads(config_path.read_text())
    split_path = ROOT / config['split_config']
    split = json.loads(split_path.read_text())
    original = json.loads((ROOT / 'configs/skillcorner_encoder_pretrain.json').read_text())
    assert split['weak_train_matches'] == original['train_matches']
    assert split['weak_development_matches'] == original['validation_matches']
    assert not set(split['weak_train_matches']) & set(split['weak_development_matches'])
    assert not config['provided_match_used'] and not config['public_labels_used']
    allowed = [REFERENCE,OLD / 'weak_preparation.json',OLD / 'weak_preparation_plan.json']
    for match in split['weak_train_matches']:
        for period in (1,2):
            stem = OLD / 'weak_features' / f'{match}_{period}_detected'
            allowed += [stem.with_suffix('.npy'),stem.with_suffix('.json'),stem.with_name(stem.name+'_weak.npy')]
    denied = restrict_external(args.out,allowed)
    preparation = json.loads((OLD / 'weak_preparation.json').read_text())
    assert preparation['complete'] and not preparation['validation_matches_used']
    assert not preparation['provided_match_used'] and not preparation['public_labels_used']
    assert not preparation['unexpected_competition_access']
    for path,digest in preparation['fingerprint']['sources'].items():
        assert sha256(ROOT / path) == digest,path
    reference = json.loads(REFERENCE.read_text())
    names = reference['names']
    assert len(names) == 1088
    import torch
    state = torch.load(ROOT / config['initial_model'] / 'selected.pt',map_location='cpu',weights_only=False)
    assert names[:664]+names[960:] == state['names']
    sources = ['scripts/research/spatial_memory/prepare_geometry_extension.py',
        'scripts/research/spatial_memory/external_guard.py','possession/skillcorner.py',
        'possession/features_v08.py','possession/features_v09.py','possession/features_v13.py',
        'possession/features_v25.py','possession/external.py']
    plan = dict(config=config,split=split,config_sha256=sha256(config_path),split_sha256=sha256(split_path),
        old_preparation_sha256=sha256(OLD / 'weak_preparation.json'),reference_schema_sha256=sha256(REFERENCE),
        sources={p:sha256(ROOT / p) for p in sources},added_names=names[664:960],full_names=names,
        old_model_sha256=sha256(ROOT / config['initial_model'] / 'selected.pt'),
        provided_match_used=False,public_labels_used=False,validation_matches_used=False)
    if (args.out / 'extension_plan.json').exists():
        assert json.loads((args.out / 'extension_plan.json').read_text()) == plan
    dump(args.out / 'extension_plan.json',plan)
    from possession.skillcorner import load_match,resample_coordinates
    from possession.features_v13 import extra_features as fine_features
    from possession.features_v25 import extra_features as tube_features
    directory = args.out / 'weak_extension'
    directory.mkdir(exist_ok=True)
    records = []
    for match in split['weak_train_matches']:
        folder = ROOT / 'data/external/skillcorner-raw' / match
        source = json.loads((folder / 'source.json').read_text(encoding='utf-8'))
        assert sha256(folder / 'metadata.json') == source['metadata_sha256']
        assert sha256(folder / 'tracking.jsonl') == source['tracking_sha256']
        metadata = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
        pitch = np.array([metadata['pitch_length'],metadata['pitch_width']],np.float32)
        halves,teams = load_match(metadata,folder / 'tracking.jsonl')
        for period in (1,2):
            started = time.monotonic()
            stem = f'{match}_{period}_detected'
            path = directory / (stem + '.npy')
            receipt_path = path.with_suffix('.json')
            old_path = OLD / 'weak_features' / (stem + '.npy')
            old_receipt = json.loads(old_path.with_suffix('.json').read_text())
            target = old_path.with_name(stem+'_weak.npy')
            assert sha256(old_path) == old_receipt['sha256'] and sha256(target) == old_receipt['target_sha256']
            assert old_receipt['fingerprint']['source'] == source
            assert old_receipt['names'] == state['names']
            fingerprint = dict(plan=plan,source=source,match=match,period=period,
                old_sha256=old_receipt['sha256'],target_sha256=old_receipt['target_sha256'],
                old_receipt_sha256=sha256(old_path.with_suffix('.json')))
            if path.exists() and receipt_path.exists():
                previous = json.loads(receipt_path.read_text())
                assert previous['fingerprint'] == fingerprint and previous['sha256'] == sha256(path)
                records.append(previous)
                print('Reused extension '+stem,flush=True)
                continue
            players,ball = halves[period]['detected']
            p,low = resample_coordinates(players+pitch/2.,speed_limit=15.)
            b,ball_low = resample_coordinates(ball+pitch/2.,speed_limit=80.)
            np.testing.assert_array_equal(low,ball_low)
            fine,fine_names = fine_features(p,teams,b,step=2)
            tube,tube_names = tube_features(p,teams,b,step=2)
            x = np.concatenate([fine,tube],axis=1)
            assert fine_names+tube_names == names[664:960]
            assert x.shape == (old_receipt['summary']['steps'],296)
            assert x.dtype == np.float32 and np.isfinite(x).all()
            temporary = path.with_suffix('.partial.npy')
            np.save(temporary,x);temporary.replace(path)
            record = dict(fingerprint=fingerprint,sha256=sha256(path),names=names[664:960],
                rows=len(x),seconds=time.monotonic()-started)
            dump(receipt_path,record)
            records.append(record)
            print(f'Prepared extension {stem}: {len(x)} rows, {record["seconds"]:.1f}s',flush=True)
            del p,b,fine,tube,x
    for path,digest in plan['sources'].items():
        assert sha256(ROOT / path) == digest,path
    dump(args.out / 'extension_preparation.json',dict(complete=True,plan=plan,records=records,
        provided_match_used=False,public_labels_used=False,validation_matches_used=False,
        weak_labels_changed=False,old_features_changed=False,unexpected_competition_access=denied))
    print(json.dumps(dict(complete=True,halves=len(records))),flush=True)


if __name__ == '__main__':
    main()
