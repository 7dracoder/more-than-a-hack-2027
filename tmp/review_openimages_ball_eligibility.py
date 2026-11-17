import csv
import json
from collections import Counter
from pathlib import Path
root=Path(__file__).resolve().parents[1];folder=root/'data/external/openimages-ball-review-20261115'
names=dict(csv.reader((folder/'classes.csv').open(encoding='utf-8')))
tree=json.loads((folder/'hierarchy.json').read_text())
def find(node,label):
    if node['LabelName']==label:return node
    return next((found for child in node.get('Subcategory',[]) if (found:=find(child,label)) is not None),None)
def descend(node):return {node['LabelName']}|set().union(*(descend(c) for c in node.get('Subcategory',[])))
ball=find(tree,'/m/018xm');classes=descend(ball)
all_boxes=list(csv.DictReader((folder/'boxes.csv').open(encoding='utf-8')))
boxes=[r for r in all_boxes if r['LabelName'] in classes]
bad={r['ImageID'] for r in boxes if int(r['IsGroupOf']) or int(r['IsDepiction'])}
good=[r for r in boxes if r['ImageID'] not in bad]
images={r['ImageID']:r for r in csv.DictReader((folder/'images.csv').open(encoding='utf-8'))}
ids={r['ImageID'] for r in good}
result=dict(ball_hierarchy={k:names[k] for k in classes},total_boxes=len(boxes),images=len({r['ImageID'] for r in boxes}),excluded_group_or_depiction_images=len(bad),remaining_images=len(ids),remaining_boxes=len(good),remaining_football_boxes=sum(r['LabelName']=='/m/01226z' for r in good),license_counts=dict(Counter(images[i]['License'] for i in ids)),authors=len({images[i]['AuthorProfileURL'] for i in ids}),image_downloads=False)
(folder/'eligibility.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
