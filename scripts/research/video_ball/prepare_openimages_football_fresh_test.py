"""Acquire an author/photo-disjoint official-test football benchmark with credits."""
import argparse
import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
from urllib.request import Request,urlopen
import numpy as np

ROOT=Path(__file__).resolve().parents[3]


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def photo_id(item):
    match=re.search(r'/([0-9]{5,})_[0-9a-f]+',item['OriginalURL'])
    return match.group(1) if match else None


def key(item):return hashlib.sha256(('coreorigin-openimages-football-v1:'+item['ImageID']).encode()).hexdigest()


def author(item):return item['AuthorProfileURL'].replace('http://','https://').rstrip('/').lower()


def split(item):
    hashed=hashlib.sha256(('coreorigin-openimages-football-v1:'+author(item)).encode()).hexdigest()
    return 'final_test' if int(hashed[:8],16)%5==0 else 'development'


def overlap(a,b):
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    return intersection/max(1e-12,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--metadata',type=Path,default=ROOT/'data/external/openimages-test-metadata-20261115');parser.add_argument('--out',type=Path,default=ROOT/'data/external/openimages-football-fresh-test-20261115');args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True);cfg=json.loads((ROOT/'configs/openimages_football_fresh_test.json').read_text())
    if (args.out/'manifest.json').exists():raise ValueError('Preserve completed acquisition')
    metadata={r['ImageID']:r for r in csv.DictReader((args.metadata/'images.csv').open(encoding='utf-8'))}
    ball_classes=set(cfg['ball_classes']);boxes={};bad=set();football=set()
    for row in csv.DictReader((args.metadata/'boxes.csv').open(encoding='utf-8')):
        if row['LabelName'] not in ball_classes:continue
        identity=row['ImageID'];box=tuple(float(row[k]) for k in ('XMin','YMin','XMax','YMax'))
        if not (0<=box[0]<box[2]<=1 and 0<=box[1]<box[3]<=1):raise ValueError('Invalid normalized box')
        boxes.setdefault(identity,{}).setdefault(box,set()).add(row['LabelName'])
        if int(row['IsGroupOf']) or int(row['IsDepiction']):bad.add(identity)
        if row['LabelName']==cfg['football_class']:football.add(identity)
    ambiguous=set()
    for identity,entries in boxes.items():
        entries=list(entries.items())
        for i,(a,labels_a) in enumerate(entries):
            for b,labels_b in entries[i+1:]:
                if labels_a!=labels_b and overlap(a,b)>.5:ambiguous.add(identity)
    positive_labels=set(boxes);negative_labels=set()
    for row in csv.DictReader((args.metadata/'human_labels.csv').open(encoding='utf-8')):
        if row['LabelName'] not in ball_classes:continue
        if float(row['Confidence'])==1:positive_labels.add(row['ImageID'])
        if row['LabelName']=='/m/018xm' and float(row['Confidence'])==0:negative_labels.add(row['ImageID'])
    coco=set(json.loads((args.metadata/'coco_photo_ids.json').read_text()));excluded={}
    prior_path=ROOT/'data/external/openimages-football-20261115/manifest.json'
    prior=json.loads(prior_path.read_text(encoding='utf-8'))
    prior_authors={r['author_group'] for r in prior['images']}
    prior_photos={r['photo_id'] for r in prior['images']}
    prior_hashes={r['sha256'] for r in prior['images']}
    eligible={}
    for identity,item in metadata.items():
        if identity not in football and identity not in negative_labels:continue
        reason=None;photo=photo_id(item)
        try:rotation=float(item['Rotation'])
        except ValueError:rotation=np.nan
        if item['License']!=cfg['license']:reason='license'
        elif not np.isfinite(rotation) or rotation!=0:reason='rotation'
        elif not item['AuthorProfileURL']:reason='author'
        elif photo is None:reason='unknown_photo_id'
        elif photo in coco:reason='coco_photo_overlap'
        elif photo in prior_photos:reason='prior_pilot_photo'
        elif author(item) in prior_authors:reason='prior_pilot_author'
        elif identity in bad:reason='group_or_depiction'
        elif identity in ambiguous:reason='ambiguous_boxes'
        if reason:excluded[reason]=excluded.get(reason,0)+1
        else:eligible[identity]=item
    positives=sorted([r for i,r in eligible.items() if i in football],key=key)
    negatives=sorted([r for i,r in eligible.items() if i in negative_labels and i not in positive_labels],key=key)
    selected=[];seen=set()
    for group,limit in [(positives,len(positives)),(negatives,cfg['negative_ratio']*len(positives))]:
        accepted=0
        for item in group:
            if photo_id(item) in seen:continue
            selected.append(dict(item,photo_id=photo_id(item),author_group=author(item),split='external_test',positive=item['ImageID'] in football,
                ball_boxes_normalized=[list(b) for b in boxes.get(item['ImageID'],{})],source_image_url=f"https://open-images-dataset.s3.amazonaws.com/test/{item['ImageID']}.jpg"))
            seen.add(photo_id(item));accepted+=1
            if accepted>=limit:break
    selected.sort(key=lambda r:r['ImageID'])
    counts={s:dict(images=sum(r['split']==s for r in selected),positive=sum(r['split']==s and r['positive'] for r in selected)) for s in ('external_test',)}
    assert all(row['positive']>=cfg['minimum_positive_images'] for row in counts.values())
    assert len({r['photo_id'] for r in selected})==len(selected)
    assert not ({r['author_group'] for r in selected}&prior_authors)
    inputs=[args.metadata/name for name in ('boxes.csv','images.csv','human_labels.csv','coco_photo_ids.json','coco_provenance.json')]
    inputs.append(prior_path)
    frozen=dict(config=cfg,config_sha256=digest(ROOT/'configs/openimages_football_fresh_test.json'),source_sha256=digest(Path(__file__)),inputs={str(p.relative_to(ROOT)):digest(p) for p in inputs},excluded=excluded,counts=counts,images=selected,provided_match_used=False)
    frozen_path=args.out/'frozen_selection.json'
    if frozen_path.exists():assert json.loads(frozen_path.read_text(encoding='utf-8'))==frozen
    else:frozen_path.write_text(json.dumps(frozen,indent=2),encoding='utf-8')
    print(json.dumps(dict(selection_frozen=True,counts=counts,excluded=excluded)),flush=True)
    images_dir=args.out/'images';images_dir.mkdir(exist_ok=True)
    def download(item):
        from PIL import Image
        identity=item['ImageID'];assert re.fullmatch('[0-9a-f]{16}',identity)
        path=images_dir/(identity+'.jpg')
        if not path.exists():
            with urlopen(Request(item['source_image_url'],headers={'User-Agent':'CoreOrigin-external-benchmark'}),timeout=60) as response:
                data=response.read(10_000_001);etag=response.headers.get('ETag','').strip('"')
            if len(data)>10_000_000:raise ValueError('Unexpected image size')
            if re.fullmatch('[0-9a-f]{32}',etag):assert hashlib.md5(data).hexdigest()==etag
            path.write_bytes(data)
        with Image.open(path) as image:
            assert image.format=='JPEG' and image.getexif().get(274,1)==1
            width,height=image.size;image.verify()
        return dict(item,sha256=digest(path),bytes=path.stat().st_size,width=width,height=height)
    with ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(download,selected))
    from collections import Counter
    occurrences=Counter(r['sha256'] for r in records)
    rejected=[r['ImageID'] for r in records if r['sha256'] in prior_hashes or occurrences[r['sha256']]>1]
    records=[r for r in records if r['ImageID'] not in rejected]
    counts={'external_test':dict(images=len(records),positive=sum(r['positive'] for r in records))}
    assert counts['external_test']['positive']>=cfg['minimum_positive_images']
    assert digest(Path(__file__))==frozen['source_sha256']
    manifest=dict(complete=True,post_download_duplicate_exclusions=rejected,prior_authors_and_photos_excluded=True,selection_sha256=digest(frozen_path),images=records,counts=counts,provided_match_used=False,annotations_license='CC BY 4.0, Google LLC; cite Kuznetsova et al. IJCV 2020.',image_license=cfg['license'],pretraining_overlap_not_fully_certified=True)
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(json.dumps(dict(complete=True,counts=counts,images=len(records))))


if __name__=='__main__':main()
