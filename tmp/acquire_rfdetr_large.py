"""Download only official public model data, pinned to immutable revisions."""
import hashlib
import json
from pathlib import Path
from urllib.request import Request,urlopen
root=Path(__file__).resolve().parents[1];repo='Roboflow/rf-detr-large'
def read(url,limit=2_000_000):
    with urlopen(Request(url,headers={'User-Agent':'CoreOrigin-external-model-review'}),timeout=60) as response:
        data=response.read(limit+1)
        if len(data)>limit:raise ValueError('Unexpected metadata size')
        return data
metadata=json.loads(read('https://huggingface.co/api/models/'+repo));sha=metadata['sha']
assert not metadata.get('gated')
tree=json.loads(read(f'https://huggingface.co/api/models/{repo}/tree/{sha}?recursive=true'))
folder=root/'data/external'/('rf-detr-large-'+sha[:12]);folder.mkdir(parents=True,exist_ok=True)
receipts={}
for name in ['config.json','preprocessor_config.json','README.md','model.safetensors']:
    entry=next(row for row in tree if row['path']==name);path=folder/name
    url=f'https://huggingface.co/{repo}/resolve/{sha}/{name}'
    if not path.exists():
        with urlopen(Request(url,headers={'User-Agent':'CoreOrigin-external-model-review'}),timeout=60) as response,path.with_suffix(path.suffix+'.partial').open('wb') as output:
            total=0
            while chunk:=response.read(1024*1024):
                total+=len(chunk)
                if total>1_000_000_000:raise ValueError('Unexpected model size')
                output.write(chunk)
        path.with_suffix(path.suffix+'.partial').replace(path)
    digest=hashlib.sha256(path.read_bytes()).hexdigest();assert path.stat().st_size==entry['size']
    if 'lfs' in entry:assert digest==entry['lfs']['oid']
    receipts[name]=dict(url=url,bytes=path.stat().st_size,sha256=digest,publisher_lfs_sha256=entry.get('lfs',{}).get('oid'))
code=json.loads(read('https://api.github.com/repos/roboflow/rf-detr/commits/develop'));code_sha=code['sha']
license_url=f'https://raw.githubusercontent.com/roboflow/rf-detr/{code_sha}/LICENSE'
license_data=read(license_url);assert b'Apache License' in license_data
(folder/'LICENSE.Apache-2.0').write_bytes(license_data)
(folder/'model_metadata.json').write_text(json.dumps(metadata,indent=2));(folder/'model_tree.json').write_text(json.dumps(tree,indent=2))
receipt=dict(repository=repo,revision=sha,folder=str(folder.relative_to(root)).replace('\\','/'),license='Apache-2.0',license_url=license_url,files=receipts,remote_code_downloaded=False,match_data_used=False,account_used=False)
(folder/'source.json').write_text(json.dumps(receipt,indent=2));(root/'data/external/rf-detr-large-source.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps({k:v for k,v in receipt.items() if k!='files'}))
