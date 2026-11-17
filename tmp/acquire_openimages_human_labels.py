import hashlib
import json
from pathlib import Path
from urllib.request import Request,urlopen
root=Path(__file__).resolve().parents[1];out=root/'data/external/openimages-ball-review-20261115';path=out/'human_labels.csv'
url='https://storage.googleapis.com/openimages/v5/validation-annotations-human-imagelabels-boxable.csv'
if not path.exists():
    with urlopen(Request(url,headers={'User-Agent':'CoreOrigin-external-label-review'}),timeout=60) as response:
        data=response.read(100_000_001)
        if len(data)>100_000_000:raise ValueError('Unexpected metadata size')
        path.write_bytes(data)
result=dict(url=url,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),provided_match_used=False)
(out/'human_labels_source.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
