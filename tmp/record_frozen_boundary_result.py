import hashlib
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1];study=root/'data/research/frozen-boundary-transfer-20261115';run=study/'run-01'
selection=json.loads((run/'selection.json').read_text());verification=json.loads((run/'verification.json').read_text())
assert selection['complete'] and verification['complete'] and verification['all_candidates_reproduced']
assert not selection['unexpected_competition_access'] and not verification['unexpected_competition_access']
assert selection['selected']['boundary_strength']==0
best=max((r for r in selection['candidates'] if r['boundary_strength']>0),key=lambda r:r['objective'])
release=json.loads((study/'source-review/review.json').read_text());assert not release['new_match_ids']
receipt=dict(complete=True,promoted=False,selected_boundary_strength=0.,retained_objective=selection['selected']['objective'],best_nonzero_strength=best['boundary_strength'],best_nonzero_objective=best['objective'],all_candidates_verified=True,backbone_frozen=True,independent_test=False,new_skillcorner_match_ids=[],competition_target_verified=False)
(run/'run_status.json').write_text(json.dumps(receipt,indent=2)+'\n')
path=root/'data/research/resumed-improvement-20261114/progress.json';progress=json.loads(path.read_text());progress['external_frozen_boundary_transfer']=receipt
progress['external_frozen_boundary_transfer']['study']='data/research/frozen-boundary-transfer-20261115'
path.write_text(json.dumps(progress,indent=2)+'\n')
files={
 'data/research/resumed-improvement-20261114/submission-ready/CoreOrigin.csv':'cbc30311a6c9fdbe8cc7198e3a25f38d444c07c437e491528da5a924ecca408d',
 'data/bar-ath-check-20261103/submission/CoreOrigin.csv':'496b42abcb0aec5ec0d4dffa6a1d8ca3c5b386dd0ccb9e4c697e877dfa1e2c80',
 'data/bar-ath-check-20261103/model/selected.pt':'d2449c5a47534c702a17853a093dee8fc60a5e90ef571e848bd9bd70d4bcfd20',
 'evaluation/run_evaluation.py':'4c09764c06768fa0e3c661a93f86a14e17389a4a01f05370c96035190a1da66d',
 'evaluation/possession_metrics.py':'32c878630c74fb52a90c301485d6dff806509d8e933b1177b6035ee8e8f42b80'}
for name,expected in files.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected
assert hashlib.sha256((root/'models/coreorigin-real-broadcast-v3/selected.pt').read_bytes()).hexdigest()==json.loads((study/'embeddings/plan.json').read_text())['model_sha256']
(run/'retained_artifact_check.json').write_text(json.dumps(dict(complete=True,sha256=files,backbone_unchanged=True),indent=2)+'\n')
with (root/'WORK_DONE.md').open('ab') as output:
    output.write(b'\n\n## 2026-11-15: whole-half frozen boundary classifier\n\nExtracted 46 external full-half recurrent feature files with frozen v3; original heads exactly reproduce ten cached development outputs. Fitted one 384-dimensional balanced logistic boundary classifier on 656928 TRAIN orientation samples, converged in 81 iterations. Known class-weight odds shift is removed. Nonzero boundary blends score 0.677658 (half) and 0.677398 (full), below unchanged v4 0.678165; no promotion or new competition export. Independent verifier reconstructs sampling, TRAIN normalization, prior correction, functional probabilities and all candidate metrics/guards. Maximum float32 probability difference 3.87e-7; two targeted tests pass. Primary SkillCorner release check still has only the same 18 TRAIN and two DEV fixtures, no new untouched matches. No supplied-match data read or scored; retained CSV, frozen v11 and official source hashes unchanged. Evidence: data/research/frozen-boundary-transfer-20261115/. Report: scripts/research/spatial_memory/FROZEN_BOUNDARY_TRANSFER.md. Goal active, BAR-ATH full-match target unverified.\n')
print(json.dumps(receipt))
