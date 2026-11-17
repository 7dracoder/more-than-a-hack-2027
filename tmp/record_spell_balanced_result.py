import hashlib
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
run=root/'data/research/spell-balanced-adaptation-20261115/run-01'
selection=json.loads((run/'selection.json').read_text());verified=json.loads((run/'external_verification.json').read_text())
assert selection['complete'] and verified['complete'] and verified['training_weights_independently_recomputed']
assert not selection['unexpected_competition_access'] and not verified['unexpected_competition_access']
history=json.loads((run/'history.json').read_text());assert history[-1]['epoch']==30
baseline=json.loads((run/'initial_reproduction.json').read_text())['baseline']
trained=[r for r in history if 'validation' in r]
best=max(trained,key=lambda r:r['validation']['objective'])
assert selection['selected']['fallback'] and best['validation']['objective']<baseline['objective']
receipt=dict(complete=True,promoted=False,reason='All evaluated trained checkpoints trail the retained external development baseline.',
    selected_fallback=True,baseline_mean=baseline['objective'],best_trained_epoch=best['epoch'],best_trained_mean=best['validation']['objective'],
    best_trained_eligible=best['validation']['eligible'],training_weight_verification_passed=True,selected_model_verification_passed=True,
    provided_match_used=False,independent_test=False,competition_target_verified=False)
(run/'run_status.json').write_text(json.dumps(receipt,indent=2)+'\n')
progress_path=root/'data/research/resumed-improvement-20261114/progress.json';progress=json.loads(progress_path.read_text())
progress['external_continuity_error_audit']={'complete':True,'study':'data/research/continuity-error-audit-20261115','masked_short_spells_matched':26,'short_gt_spells':124,'iou':.5,'provided_match_used':False}
progress['external_spell_balanced_adaptation']=receipt
progress['external_football_audio_review']={'complete':True,'study':'data/research/football-audio-source-review-20261115','candidate':'WhissleAI/speech-soccer-broadcast','revision':'c686459b425888ce65c8db9cec038b443784d831','metadata_rows':327,'whistle_event_annotations':False,'possession_state_annotations':False,'audio_downloaded':False}
progress_path.write_text(json.dumps(progress,indent=2)+'\n')
files={
 'data/research/resumed-improvement-20261114/submission-ready/CoreOrigin.csv':'cbc30311a6c9fdbe8cc7198e3a25f38d444c07c437e491528da5a924ecca408d',
 'data/bar-ath-check-20261103/submission/CoreOrigin.csv':'496b42abcb0aec5ec0d4dffa6a1d8ca3c5b386dd0ccb9e4c697e877dfa1e2c80',
 'data/bar-ath-check-20261103/model/selected.pt':'d2449c5a47534c702a17853a093dee8fc60a5e90ef571e848bd9bd70d4bcfd20',
 'evaluation/run_evaluation.py':'4c09764c06768fa0e3c661a93f86a14e17389a4a01f05370c96035190a1da66d',
 'evaluation/possession_metrics.py':'32c878630c74fb52a90c301485d6dff806509d8e933b1177b6035ee8e8f42b80'}
for name,expected in files.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected
(run/'retained_artifact_check.json').write_text(json.dumps(dict(complete=True,sha256=files),indent=2)+'\n')
with (root/'WORK_DONE.md').open('ab') as output:
    entry='\n\n## 2026-11-15: equal-run weighting external control\n\nExternal v4 error audit reproduced all scores and found masked short possessions matched 26/124 at IoU 0.5, versus 126/168 for spells >=5 seconds. Completed fixed 30-epoch equal-run strong TRAIN weighting experiment with all other occurrence-safe geometry control settings unchanged. Selected initial fallback; best trained epoch '+str(best['epoch'])+' mean '+str(best['validation']['objective'])+' versus retained '+str(baseline['objective'])+'. Independent weight/normalization and selected-score verifier passes. Eight focused geometry tests pass. Public football audio metadata review found 327 ASR-only snippets, no explicit whistle/state annotations; no audio downloaded. No supplied-match training/scoring/inspection. Retained CSV, frozen v11 and official source hashes unchanged. See scripts/research/spatial_memory/SPELL_BALANCED_ADAPTATION.md and ignored data/research/spell-balanced-adaptation-20261115/run-01/. Goal active, BAR-ATH target unverified.\n'
    output.write(entry.encode('ascii'))
print(json.dumps(receipt))
