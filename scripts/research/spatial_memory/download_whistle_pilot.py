"""Acquire a bounded pinned external audio pilot and upstream model source notices."""
import hashlib
import json
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor
import requests

ROOT=Path(__file__).resolve().parents[3]


def download(url,path,expected=None,algorithm='sha256'):
    if path.exists():
        digest=hashlib.new(algorithm,path.read_bytes()).hexdigest()
        if expected:assert digest==expected
        return dict(path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),url=url)
    path.parent.mkdir(parents=True,exist_ok=True);part=path.with_suffix(path.suffix+'.partial')
    response=requests.get(url,stream=True,timeout=(25,60));response.raise_for_status();h=hashlib.new(algorithm);sha=hashlib.sha256();size=0
    with part.open('wb') as f:
        for block in response.iter_content(1024*1024):
            if block:f.write(block);h.update(block);sha.update(block);size+=len(block)
    if expected:assert h.hexdigest()==expected,(path.name,'checksum differs')
    part.replace(path)
    print('Acquired '+path.name,flush=True)
    return dict(path=str(path.relative_to(ROOT)),bytes=size,sha256=sha.hexdigest(),url=url)


def main():
    config=json.loads((ROOT/'configs/audio_whistle_pilot.json').read_text())
    folder=ROOT/'data/external/volleyball-whistles-5f23e58f6057'
    metadata=json.loads((folder/'acquisition_metadata.json').read_text())
    assert metadata['revision']==config['revision'] and metadata['declared_license']=='MIT' and not metadata['gated']
    for name,record in metadata['files'].items():assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==record['sha256']
    jobs=[]
    for match in config['calibration']+config['development']+config['final_test']:
        name=match+'.wav';record=next(x for x in metadata['tree'] if x['path']==name)
        jobs.append((f"https://huggingface.co/datasets/{config['dataset']}/resolve/{config['revision']}/{name}",folder/name,record['lfs']['oid'],'sha256'))
    repo='https://api.github.com/repos/qiuqiangkong/audioset_tagging_cnn'
    response=requests.get(repo+'/commits/master',timeout=25);response.raise_for_status();commit=response.json()['sha']
    source=ROOT/'data/external'/('panns-'+commit[:12]);source.mkdir(parents=True,exist_ok=True)
    for name in ['pytorch/models.py','pytorch/pytorch_utils.py','LICENSE.MIT','README.md','metadata/class_labels_indices.csv']:
        jobs.append((f'https://raw.githubusercontent.com/qiuqiangkong/audioset_tagging_cnn/{commit}/{name}',source/name,None,'sha256'))
    jobs.append(('https://zenodo.org/records/3987831/files/Cnn14_DecisionLevelMax_mAP%3D0.385.pth?download=1',source/'Cnn14_DecisionLevelMax.pth',config['model_md5'],'md5'))
    with ThreadPoolExecutor(max_workers=2) as pool:records=list(pool.map(lambda x:download(*x),jobs))
    result=dict(complete=True,panns_commit=commit,panns_source=str(source.relative_to(ROOT)),dataset_revision=config['revision'],files=records,
        dataset_declared_license='MIT',panns_code_license='MIT',panns_weights_license='CC BY 4.0',provided_match_used=False)
    (folder/'pilot_acquisition.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(complete=True,files=len(records),bytes=sum(r['bytes'] for r in records),panns_source=result['panns_source'])),flush=True)


if __name__=='__main__':main()
