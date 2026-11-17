import json
import hashlib
from pathlib import Path

root=Path(__file__).resolve().parents[1]
study=root/'data/research/whistle-transfer-20261115'
final=json.loads((study/'fresh-final/results.json').read_text())
assert final['complete'] and not final['unexpected_competition_access']
path=root/'data/research/resumed-improvement-20261114/progress.json'
progress=json.loads(path.read_text())
progress['external_whistle_transfer']={
    'complete':True,'study':str(study.relative_to(root)).replace('\\','/'),
    'development_whistle_event_f1':0.7416563658838071,
    'fresh_final_recording':final['match'],'fresh_final_whistle_events':final['final_test'],
    'maximum_cardinality_matching_verified':True,'provided_audio_decoded':False,
    'integrated_into_possession':False,'promoted':False,
    'next_step':'Obtain permitted external football audio with event/state evidence; test transfer and clock alignment before possession integration.'}
progress['competition_target_verified']=False
path.write_text(json.dumps(progress,indent=2)+'\n')
files={
 'data/research/resumed-improvement-20261114/submission-ready/CoreOrigin.csv':'cbc30311a6c9fdbe8cc7198e3a25f38d444c07c437e491528da5a924ecca408d',
 'data/bar-ath-check-20261103/submission/CoreOrigin.csv':'496b42abcb0aec5ec0d4dffa6a1d8ca3c5b386dd0ccb9e4c697e877dfa1e2c80',
 'data/bar-ath-check-20261103/model/selected.pt':'d2449c5a47534c702a17853a093dee8fc60a5e90ef571e848bd9bd70d4bcfd20',
 'evaluation/run_evaluation.py':'4c09764c06768fa0e3c661a93f86a14e17389a4a01f05370c96035190a1da66d',
 'evaluation/possession_metrics.py':'32c878630c74fb52a90c301485d6dff806509d8e933b1177b6035ee8e8f42b80'}
for name,expected in files.items():
    actual=hashlib.sha256((root/name).read_bytes()).hexdigest()
    assert actual==expected,(name,actual)
receipt=dict(complete=True,retained_submission_and_frozen_v11_unchanged=True,sha256=files,competition_target_verified=False)
(study/'retained_artifact_check.json').write_text(json.dumps(receipt,indent=2)+'\n')
with (root/'WORK_DONE.md').open('ab') as output:
    output.write(b'\n\n## 2026-11-15: external frozen-feature whistle adaptation\n\n'
      b'Trained a 6144-feature linear whistle head on recording 1, selected threshold 0.9 on recording 2, and froze before fresh recording 4. Fresh whistle-event F1 0.779964 (218 TP, 34 FP, 89 FN); precision 0.865079, recall 0.710098. Independent maximum matching confirms counts. Six event/target/mapping tests pass. Float32 functional verification differs by at most 3.58e-7 with unchanged development event metrics. No recording-3 retest, competition data reads, target audio decoding, or possession integration. This is not BAR-ATH spell F1. v4 CSV and frozen v11 hashes unchanged; exact retained 199319-row CSV passes official format validator. Evidence: data/research/whistle-transfer-20261115/. See scripts/research/spatial_memory/WHISTLE_TRANSFER.md. Goal remains active; football audio transfer needs external evidence.\n')
print(json.dumps(receipt))
