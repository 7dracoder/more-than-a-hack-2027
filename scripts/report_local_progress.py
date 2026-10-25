"""Read the experiment evidence ledger without assuming unfinished jobs are running."""
import argparse
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def verified_target(run,selection):
    if selection['validation']['spell_f1_mean']<.9:
        return False
    try:
        def digest(path):
            with path.open('rb') as file:return hashlib.file_digest(file,'sha256').hexdigest()
        audit=json.loads((run/'campaign-audit/audit.json').read_text())
        receipt=json.loads((run/'reproducibility.json').read_text())
        checkpoint=digest(run/'selected.pt')
        candidate=digest(run/'candidate/CoreOrigin.csv')
        first=digest(run/'candidate/CoreOrigin.first-pass.csv')
        return (checkpoint==selection['checkpoint_sha256']==audit['selected_checkpoint_sha256']==receipt['selected_sha256']
                and audit['scores']['spell_f1_mean']==selection['validation']['spell_f1_mean']
                and receipt['two_complete_passes_byte_identical'] and receipt['rows']==199319
                and candidate==first==receipt['sha256'])
    except (OSError,KeyError,ValueError):
        return False


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT/'data/local-v08')
    args=p.parse_args()
    rows=[]
    for run in sorted(args.root.glob('run-*')):
        if not run.is_dir(): continue
        row=dict(run=run.name,status='unfinished')
        if (run/'failure.json').exists():
            row.update(json.loads((run/'failure.json').read_text()))
        elif (run/'selection.json').exists():
            selection=json.loads((run/'selection.json').read_text())
            row.update(status='fit_complete',score=selection['validation']['spell_f1_mean'],validation=selection['validation'])
            row['target_verified']=verified_target(run,selection)
        elif (run/'history.json').exists():
            history=json.loads((run/'history.json').read_text())
            scored=[r for r in history if 'validation' in r]
            row['last_completed_epoch']=history[-1]['epoch'] if history else None
            if scored:
                best=max(scored,key=lambda r:r['validation']['spell_f1_mean'])
                row.update(score=best['validation']['spell_f1_mean'],best_epoch=best['epoch'])
        row['candidate_exists']=(run/'candidate/CoreOrigin.csv').exists()
        rows.append(row)
    completed=[r for r in rows if r.get('status')=='fit_complete']
    best=max(completed,key=lambda r:r['score']) if completed else None
    report=dict(scope='Fixed external development experiments only; this ledger arms no goal or automation',
                target_spell_f1=.9,best_completed=best,target_fit_found=best is not None and best['score']>=.9,
                target_met=any(r.get('target_verified',False) for r in completed),
                current_independent_score=None,official_competition_score=None,runs=rows)
    (args.root/'progress.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(best_completed=best,target_met=report['target_met'],runs=len(rows)),indent=2))


if __name__=='__main__': main()
