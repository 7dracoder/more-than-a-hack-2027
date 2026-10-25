"""Run a finite, predeclared local experiment batch sequentially on one GPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    with open(path,'rb') as file:
        return hashlib.file_digest(file,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,default=ROOT/'configs/development_campaign_v13.json')
    args=parser.parse_args(); plan=json.loads(args.plan.read_text())
    ledger=ROOT/'data/local-v08/campaign_status.json'
    state=dict(plan=str(args.plan),plan_sha256=digest(args.plan),scope=plan['scope'],target=plan['target_spell_f1'],status='waiting',completed=[])
    def update(**values):
        state.update(values)
        partial=ledger.with_suffix('.partial.json');partial.write_text(json.dumps(state,indent=2));partial.replace(ledger)
        print(json.dumps(values),flush=True)
    def wait_for(paths):
        deadline=time.monotonic()+3600
        while not all(path.exists() for path in paths):
            if time.monotonic()>deadline:
                raise TimeoutError('Local dependency did not finish within one hour')
            time.sleep(5)
    def command(arguments,log):
        with log.open('w') as file:
            subprocess.run([sys.executable,*arguments],cwd=ROOT,env=os.environ,stdout=file,stderr=subprocess.STDOUT,check=True)
    def certify(run):
        # A new score is accepted only after fresh inference and unchanged scoring.
        audit=run/'campaign-audit'
        command(['scripts/audit_local_errors.py','--run',str(run),'--out',str(audit)],run/'campaign-audit.log')
        fresh=json.loads((audit/'audit.json').read_text())['scores']['spell_f1_mean']
        selected=json.loads((run/'selection.json').read_text())['validation']['spell_f1_mean']
        if fresh!=selected:
            raise ValueError('Fresh development audit disagrees with selected checkpoint')
        command(['scripts/evaluate_gap_stress.py','--run',str(run)],run/'campaign-stress.log')
        return fresh
    def export_repeat(run,config):
        path=run/'candidate/CoreOrigin.csv'; first=run/'candidate/CoreOrigin.first-pass.csv'
        receipt_path=run/'reproducibility.json'
        if receipt_path.exists() and path.exists() and first.exists():
            receipt=json.loads(receipt_path.read_text())
            if receipt.get('two_complete_passes_byte_identical') and receipt['sha256']==digest(path)==digest(first) and receipt['selected_sha256']==digest(run/'selected.pt'):
                return
            raise ValueError('Existing repeat receipt no longer matches frozen artifacts')
        if not path.exists():
            command(['scripts/run_local.py','--infer','--config',str(config),'--out',str(run)],run/'campaign-export-first.log')
        if not first.exists(): shutil.copyfile(path,first)
        command(['scripts/run_local.py','--infer','--config',str(config),'--out',str(run)],run/'campaign-export-repeat.log')
        if digest(path)!=digest(first):
            raise ValueError('Two full tactical exports differ')
        receipt=dict(two_complete_passes_byte_identical=True,sha256=digest(path),
                     selected_sha256=digest(run/'selected.pt'),rows=199319,official_competition_score=None)
        (run/'reproducibility.json').write_text(json.dumps(receipt,indent=2))
    try:
        update(status='waiting_for_active_fit')
        wait_for([ROOT/plan['after_run']/'selection.json'])
        best=ROOT/plan['initial_best']
        best_score=json.loads((best/'selection.json').read_text())['validation']['spell_f1_mean']
        after=ROOT/plan['after_run']
        after_score=json.loads((after/'selection.json').read_text())['validation']['spell_f1_mean']
        if after_score>best_score:
            update(status='auditing_active_fit_improvement',current_run=str(after),candidate_score=after_score)
            certify(after);export_repeat(after,ROOT/plan['after_config'])
            best,best_score=after,after_score
            update(best_run=str(best),best_score=best_score,target_met=best_score>=plan['target_spell_f1'])
            if best_score>=plan['target_spell_f1']:
                update(status='target_met_on_development_only');return
        update(status='repeating_existing_export',best_run=str(best),best_score=best_score)
        if best!=after:
            export_repeat(best,ROOT/plan.get('initial_config','configs/local_v10_fullrate.json'))
        feature=json.loads((ROOT/plan['wait_for_feature_config']).read_text())
        ids=feature['train_matches']+feature['validation_matches']
        wait_for([ROOT/directory/(match+'.json') for directory in (feature['features_dir'],feature['augmentation_features_dir']) for match in ids])
        for job in plan['runs']:
            run=ROOT/job['out']; config=ROOT/job['config']
            if run.exists(): raise ValueError('Campaign will not overwrite an existing run: '+str(run))
            update(status='training',current_run=str(run),config=str(config))
            command(['scripts/run_local.py','--train','--config',str(config),'--out',str(run)],run.with_suffix('.log'))
            command(['scripts/snapshot_local_run.py','--run',str(run)],run/'snapshot.log')
            selection=json.loads((run/'selection.json').read_text())
            score=selection['validation']['spell_f1_mean']
            state['completed'].append(dict(run=str(run),score=score,checkpoint_sha256=selection['checkpoint_sha256']))
            if score>best_score:
                update(status='auditing_improvement',current_run=str(run),candidate_score=score)
                certify(run);export_repeat(run,config)
                best,best_score=run,score
            command(['scripts/report_local_progress.py'],ROOT/'data/local-v08/campaign-progress.log')
            update(best_run=str(best),best_score=best_score,target_met=best_score>=plan['target_spell_f1'])
            if best_score>=plan['target_spell_f1']:
                update(status='target_met_on_development_only');return
        update(status='planned_runs_complete_target_unmet',target_met=False)
    except Exception as error:
        update(status='failed',error=repr(error));raise


if __name__=='__main__': main()
