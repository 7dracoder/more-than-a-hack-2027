"""Public unauthenticated metadata only; no account, audio or competition data."""
import hashlib
import json
from pathlib import Path
from urllib.request import Request,urlopen

root=Path(__file__).resolve().parents[1]
out=root/'data/research/football-audio-source-review-20261115';out.mkdir(parents=True,exist_ok=True)
base='https://huggingface.co';dataset='WhissleAI/speech-soccer-broadcast'
def get(url):
    with urlopen(Request(url,headers={'User-Agent':'CoreOrigin-external-metadata-review'}),timeout=60) as response:return response.read()
metadata=json.loads(get(f'{base}/api/datasets/{dataset}'))
revision=metadata['sha']
tree=json.loads(get(f'{base}/api/datasets/{dataset}/tree/{revision}?recursive=true'))
card=get(f'{base}/datasets/{dataset}/resolve/{revision}/README.md')
(out/'metadata.json').write_text(json.dumps(metadata,indent=2))
(out/'tree.json').write_text(json.dumps(tree,indent=2))
(out/'README.source.md').write_bytes(card)
result=dict(dataset=dataset,revision=revision,gated=metadata.get('gated'),card_bytes=len(card),card_sha256=hashlib.sha256(card).hexdigest(),
    files=[dict(path=r['path'],size=r.get('size'),type=r['type']) for r in tree],audio_downloaded=False,account_used=False)
(out/'review.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
