"""Create the disjoint official-test acquisition runner without changing the pilot."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
old_metadata=root/'data/external/openimages-ball-review-20261115';new_metadata=root/'data/external/openimages-test-metadata-20261115'
for name in ('coco_photo_ids.json','coco_provenance.json'):
    source=(old_metadata/name).read_bytes();target=new_metadata/name
    if target.exists():assert target.read_bytes()==source
    else:target.write_bytes(source)
cfg=json.loads((root/'configs/openimages_football_detector.json').read_text())
cfg.update(source='Open Images official TEST split, V5 boxes/human labels and 2018_04 image metadata.',
    minimum_positive_images=20,split='All eligible records are external_test; exclude every prior pilot author and photo identity.',
    qualification=cfg['qualification']+' Also exclude prior pilot authors/photos and exact JPEG duplicates against the pilot or within this cohort before inference.',
    selection_metric='No selection on this corpus; validate the previously frozen development-selected tiny-ball method and frozen controls once.')
(root/'configs/openimages_football_fresh_test.json').write_text(json.dumps(cfg,indent=2)+'\n')
source=root/'scripts/research/video_ball/prepare_openimages_football.py';target=source.with_name('prepare_openimages_football_fresh_test.py');assert not target.exists()
text=source.read_text()
text=text.replace('Freeze and acquire an external football detector benchmark with credits.','Acquire an author/photo-disjoint official-test football benchmark with credits.')
text=text.replace('data/external/openimages-ball-review-20261115','data/external/openimages-test-metadata-20261115')
text=text.replace('data/external/openimages-football-20261115','data/external/openimages-football-fresh-test-20261115')
text=text.replace('configs/openimages_football_detector.json','configs/openimages_football_fresh_test.json')
needle="    coco=set(json.loads((args.metadata/'coco_photo_ids.json').read_text()));excluded={}\n"
assert text.count(needle)==1
text=text.replace(needle,needle+'''    prior_path=ROOT/'data/external/openimages-football-20261115/manifest.json'
    prior=json.loads(prior_path.read_text(encoding='utf-8'))
    prior_authors={r['author_group'] for r in prior['images']}
    prior_photos={r['photo_id'] for r in prior['images']}
    prior_hashes={r['sha256'] for r in prior['images']}
''')
text=text.replace("        elif photo in coco:reason='coco_photo_overlap'", "        elif photo in coco:reason='coco_photo_overlap'\n        elif photo in prior_photos:reason='prior_pilot_photo'\n        elif author(item) in prior_authors:reason='prior_pilot_author'")
text=text.replace('split=split(item)',"split='external_test'")
text=text.replace("('development','final_test')","('external_test',)")
text=text.replace("assert all(row['positive']>=5 for row in counts.values())","assert all(row['positive']>=cfg['minimum_positive_images'] for row in counts.values())")
text=text.replace("    assert not ({r['author_group'] for r in selected if r['split']=='development'}&{r['author_group'] for r in selected if r['split']=='final_test'})","    assert not ({r['author_group'] for r in selected}&prior_authors)")
text=text.replace("    frozen=dict(config=cfg,", "    inputs.append(prior_path)\n    frozen=dict(config=cfg,")
text=text.replace(".s3.amazonaws.com/validation/", ".s3.amazonaws.com/test/")
needle="    assert digest(Path(__file__))==frozen['source_sha256']\n"
assert text.count(needle)==1
text=text.replace(needle,'''    from collections import Counter
    occurrences=Counter(r['sha256'] for r in records)
    rejected=[r['ImageID'] for r in records if r['sha256'] in prior_hashes or occurrences[r['sha256']]>1]
    records=[r for r in records if r['ImageID'] not in rejected]
    counts={'external_test':dict(images=len(records),positive=sum(r['positive'] for r in records))}
    assert counts['external_test']['positive']>=cfg['minimum_positive_images']
'''+needle)
text=text.replace('manifest=dict(complete=True,','manifest=dict(complete=True,post_download_duplicate_exclusions=rejected,prior_authors_and_photos_excluded=True,')
target.write_text(text)
print('Created immutable-pilot-preserving fresh-test runner')
