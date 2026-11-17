import hashlib
import json
from pathlib import Path
from urllib.request import Request,urlopen
root=Path(__file__).resolve().parents[1];out=root/'data/external/openimages-test-metadata-20261115';out.mkdir(parents=True,exist_ok=True)
sources={
 'images.csv':'https://storage.googleapis.com/openimages/2018_04/test/test-images-with-rotation.csv',
 'boxes.csv':'https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv',
 'human_labels.csv':'https://storage.googleapis.com/openimages/v5/test-annotations-human-imagelabels-boxable.csv'}
receipt={}
for name,url in sources.items():
    path=out/name
    if not path.exists():
        partial=path.with_suffix('.partial')
        with urlopen(Request(url,headers={'User-Agent':'CoreOrigin-external-test-review'}),timeout=60) as response,partial.open('wb') as output:
            total=0
            while chunk:=response.read(1024*1024):
                total+=len(chunk)
                if total>150_000_000:raise ValueError('Unexpected metadata size')
                output.write(chunk)
        partial.replace(path)
    receipt[name]=dict(url=url,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
(out/'source.json').write_text(json.dumps(dict(files=receipt,provided_match_used=False,images_downloaded=False),indent=2));print(json.dumps(receipt))
