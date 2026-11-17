"""Acquire public external metadata only and inspect ball-class eligibility."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from urllib.request import Request,urlopen
root=Path(__file__).resolve().parents[1];out=root/'data/external/openimages-ball-review-20261115';out.mkdir(parents=True,exist_ok=True)
urls={
 'classes.csv':'https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv',
 'boxes.csv':'https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv',
 'images.csv':'https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv',
 'hierarchy.json':'https://storage.googleapis.com/openimages/2018_04/bbox_labels_600_hierarchy.json'}
receipts={}
for name,url in urls.items():
    path=out/name
    if not path.exists():
        with urlopen(Request(url,headers={'User-Agent':'CoreOrigin-external-metadata-review'}),timeout=60) as response:
            data=response.read(100_000_001)
            if len(data)>100_000_000:raise ValueError('Unexpected metadata size')
            path.write_bytes(data)
    receipts[name]=dict(url=url,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
classes=dict(csv.reader((out/'classes.csv').open(encoding='utf-8')))
ball_classes={key:name for key,name in classes.items() if 'ball' in name.lower()}
boxes=list(csv.DictReader((out/'boxes.csv').open(encoding='utf-8')))
images=list(csv.DictReader((out/'images.csv').open(encoding='utf-8')))
counts={key:dict(name=name,boxes=sum(r['LabelName']==key for r in boxes),images=len({r['ImageID'] for r in boxes if r['LabelName']==key})) for key,name in ball_classes.items()}
result=dict(complete=True,receipts=receipts,ball_classes=counts,image_fields=list(images[0]),licenses=dict(Counter(r.get('License','') for r in images)),images_downloaded=False,provided_match_used=False)
(out/'review.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
