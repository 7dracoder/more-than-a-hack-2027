import hashlib
import json
from collections import Counter
from pathlib import Path
from urllib.request import Request,urlopen
root=Path(__file__).resolve().parents[1];out=root/'data/research/football-audio-source-review-20261115'
review=json.loads((out/'review.json').read_text());revision=review['revision'];dataset=review['dataset']
url=f'https://huggingface.co/datasets/{dataset}/resolve/{revision}/processed_soccer_files.jsonl'
with urlopen(Request(url,headers={'User-Agent':'CoreOrigin-external-schema-review'}),timeout=60) as response:raw=response.read()
(out/'processed_soccer_files.source.jsonl').write_bytes(raw)
rows=[json.loads(line) for line in raw.decode().splitlines() if line.strip()]
fields=Counter();types={}
for row in rows:
    for key,value in row.items():
        fields[key]+=1;types.setdefault(key,set()).add(type(value).__name__)
result=dict(revision=revision,rows=len(rows),sha256=hashlib.sha256(raw).hexdigest(),fields=dict(fields),types={k:sorted(v) for k,v in types.items()},audio_downloaded=False,transcript_contents_not_printed=True)
(out/'schema.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
