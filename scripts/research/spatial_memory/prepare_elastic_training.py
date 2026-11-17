"""Pin only two already-designated TRAIN fixtures' external event annotations."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
COMMIT = 'e245224aad7450961eca40954ae2bc85e1e6e696'
TRAIN = ['J03WMX','J03WN1']
REPO = 'https://api.github.com/repos/hyunsungkim-ds/elastic'
RAW = f'https://raw.githubusercontent.com/hyunsungkim-ds/elastic/{COMMIT}/'
FILES = ['README.md','LICENSE']+[f'benchmark/gt/{match}.parquet' for match in TRAIN]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to((ROOT/'data/external').resolve()):
        raise ValueError('Download only into local ignored external-data storage')
    split = json.loads((ROOT/'configs/real_broadcast_memory_adaptation.json').read_text())
    assert set(TRAIN)<=set(split['strong_train_matches'])
    assert not set(TRAIN)&set(split['strong_development_matches'])
    assert all('WPY' not in p and 'BAR' not in p and 'ATH' not in p for p in FILES)
    data,truth = (ROOT/'data').resolve(),(ROOT/'evaluation/gt').resolve()
    denied = []

    def guard(event,values):
        if event!='open' or not values or not isinstance(values[0],(str,bytes,os.PathLike)):
            return
        path = Path(os.fsdecode(values[0])).resolve()
        if path.is_relative_to(truth) or (path.is_relative_to(data) and not path.is_relative_to(out)):
            denied.append(str(path))
            raise PermissionError('External acquisition cannot open other data or possession truth')

    sys.addaudithook(guard)
    out.mkdir(parents=True,exist_ok=True)
    receipt = out/'source.json'
    if receipt.exists():
        prior = json.loads(receipt.read_text())
        assert prior['complete'] and prior['commit']==COMMIT
        for name,record in prior['files'].items():
            assert digest(out/name)==record['sha256']
        print('Pinned training-only external annotations already verified.')
        return
    import requests
    import pandas as pd
    session = requests.Session()
    session.headers['User-Agent'] = 'CoreOrigin-external-training-provenance'
    response = session.get(REPO+'/git/trees/'+COMMIT,params={'recursive':1},timeout=25)
    response.raise_for_status()
    tree = {v['path']:v for v in response.json()['tree'] if v['type']=='blob'}
    records = {}
    for name in FILES:
        response = session.get(RAW+name,timeout=25)
        response.raise_for_status()
        content = response.content
        blob = hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
        assert len(content)==tree[name]['size'] and blob==tree[name]['sha'],name
        path = out/name
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():
            assert path.read_bytes()==content,'Preserve an existing acquisition'
        else:
            path.write_bytes(content)
        records[name] = dict(url=RAW+name,bytes=len(content),git_blob_sha=blob,sha256=digest(path))
    readme = (out/'README.md').read_text(encoding='utf-8')
    assert 'CC BY 4.0' in readme and 'MPL-2.0' in readme and 're-annotated timestamps' in readme
    schemas = {}
    for match in TRAIN:
        frame = pd.read_parquet(out/f'benchmark/gt/{match}.parquet')
        assert {'frame_id','receive_frame_id'}<=set(frame.columns)
        schemas[match] = dict(rows=len(frame),columns={str(k):str(v) for k,v in frame.dtypes.items()},
            timestamp_missing=int(frame.frame_id.isna().sum()),reception_timestamp_present=int(frame.receive_frame_id.notna().sum()))
    record = dict(complete=True,repository='https://github.com/hyunsungkim-ds/elastic',commit=COMMIT,
        data_license='CC BY 4.0',code_license='MPL-2.0',
        attribution='Kim et al., ELASTIC, CIKM 2026; Bassek et al., Scientific Data 12, 195 (2025); Deutsche Fussball Liga (DFL).',
        matches=TRAIN,existing_train_matches_only=True,new_independent_matches=0,files=records,schemas=schemas,
        downloaded_development_annotations=False,provided_match_used=False,public_labels_used=False,
        used_in_current_model=False,unexpected_competition_access=denied,source_sha256=digest(Path(__file__)))
    receipt.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(complete=True,commit=COMMIT,schemas=schemas,downloaded_development_annotations=False)))


if __name__=='__main__':
    main()
