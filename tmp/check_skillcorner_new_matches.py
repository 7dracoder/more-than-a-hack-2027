"""Check public release metadata for genuinely new external match identities."""
import json
from pathlib import Path
from urllib.request import Request,urlopen
root=Path(__file__).resolve().parents[1]
out=root/'data/research/frozen-boundary-transfer-20261115/source-review';out.mkdir(parents=True,exist_ok=True)
def get(url):
    with urlopen(Request(url,headers={'User-Agent':'CoreOrigin-external-release-review'}),timeout=60) as response:return json.loads(response.read())
commit=get('https://api.github.com/repos/SkillCorner/opendata/commits/master');sha=commit['sha']
matches=get(f'https://raw.githubusercontent.com/SkillCorner/opendata/{sha}/data/matches.json')
cfg=json.loads((root/'configs/real_broadcast_memory_adaptation.json').read_text());known=set(cfg['weak_train_matches']+cfg['weak_development_matches']);released={str(row['id']) for row in matches}
(out/'commit.json').write_text(json.dumps(commit,indent=2));(out/'matches.json').write_text(json.dumps(matches,indent=2))
result=dict(repository='https://github.com/SkillCorner/opendata',revision=sha,released_matches=sorted(released),new_match_ids=sorted(released-known),existing_training_matches=len(released&set(cfg['weak_train_matches'])),existing_development_matches=len(released&set(cfg['weak_development_matches'])),tracking_downloaded=False)
(out/'review.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
