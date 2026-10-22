"""Summarize completed v0.4 experiments without refitting or changing predictions."""
import html
import json
from pathlib import Path
import platform
import sys
import argparse
import math

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/experiments/blindspot-v04'


def main(OUT=OUT):
    if not (OUT / 'environment.json').exists():
        import numpy, pandas, scipy, sklearn, joblib
        environment = {'python': sys.version, 'platform': platform.platform(), 'threads': 2,
                       'packages': {m.__name__: m.__version__ for m in [numpy, pandas, scipy, sklearn, joblib]}}
        (OUT / 'environment.json').write_text(json.dumps(environment, indent=2), encoding='utf-8')
    scores = json.loads((OUT / 'scores.json').read_text())
    baseline = scores['baseline']
    eligible = [name for name, score in scores.items() if name != 'baseline' and all(
        score[p]['spell'] - baseline[p]['spell'] > .06 for p in ('whole_run', 'fixed_time'))]
    chosen = max(eligible, key=lambda name: min(scores[name][p]['spell'] - baseline[p]['spell']
                  for p in ('whole_run', 'fixed_time'))) if eligible else 'baseline'
    selection = {'provisional_group': chosen, 'eligible': eligible,
                 'rule': 'Gain greater than 0.06 spell F1 in both protocols; rank eligible groups by smaller of the two gains.',
                 'scope': 'Conservative development heuristic, not a significance test. Selection reuses the inspected public sample; no independent-match validation.'}
    (OUT / 'selection.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')
    diagnostics = {}
    for name in scores:
        diagnostics[name] = {}
        for protocol in ('whole_run', 'fixed_time'):
            folds = json.loads((OUT / name / f'{protocol}_folds.json').read_text())
            cm = [[sum(f['score']['confusion'][i][j] for f in folds) for j in range(4)] for i in range(4)]
            transitions = {}
            for tau in ('12', '25'):
                matched = sum(f['score']['transitions'][tau]['n_matched'] for f in folds)
                missed = sum(f['score']['transitions'][tau]['n_missed'] for f in folds)
                extra = sum(f['score']['transitions'][tau]['n_spurious'] for f in folds)
                transitions[tau] = {'matched': matched, 'missed': missed, 'extra': extra,
                                    'f1': 2*matched/(2*matched+missed+extra) if 2*matched+missed+extra else 0.}
            diagnostics[name][protocol] = {'wrong_live_team_frames': cm[0][1]+cm[1][0],
                'live_called_dead_frames': cm[0][2]+cm[1][2],
                'dead_called_live_frames': cm[2][0]+cm[2][1],
                'abstained_graded_frames': sum(cm[i][3] for i in range(3)),
                'confusion': cm, 'transitions': transitions}
    (OUT / 'diagnostics.json').write_text(json.dumps(diagnostics, indent=2), encoding='utf-8')
    head_note = ''
    if 'contacts_team_only' in scores:
        checks = []
        for protocol in ('whole_run', 'fixed_time'):
            baseline_folds = json.loads((OUT / 'baseline' / f'{protocol}_folds.json').read_text())
            contact_folds = json.loads((OUT / 'contacts' / f'{protocol}_folds.json').read_text())
            hybrid_folds = json.loads((OUT / 'contacts_team_only' / f'{protocol}_folds.json').read_text())
            for baseline_fold, contact_fold, hybrid_fold in zip(baseline_folds, contact_folds, hybrid_folds):
                audit = hybrid_fold['model_audit']
                same_masks = (audit['baseline_model']['training_mask_sha256'] == audit['contacts_model']['training_mask_sha256']
                              == baseline_fold['train_mask_sha256'] == contact_fold['train_mask_sha256'] == hybrid_fold['train_mask_sha256'])
                if not (same_masks and audit['baseline_dead_mask_unchanged'] and audit['baseline_ungraded_mask_unchanged']):
                    raise ValueError('Hybrid head isolation check failed')
                checks.append({'protocol': protocol, 'fold': hybrid_fold['fold'], 'identical_training_masks': same_masks,
                               'dead_mask_unchanged': True, 'ungraded_mask_unchanged': True})
            if not math.isclose(scores['baseline'][protocol]['dead'], scores['contacts_team_only'][protocol]['dead'], abs_tol=1e-12):
                raise ValueError('Hybrid changed dead-ball F1')
        (OUT / 'head_invariants.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
        head_note = ('Team-only contacts uses baseline P(D) and contacts conditional H/A evidence. Both classifiers use identical training masks. '
                     'Decoded dead-ball and ungraded masks are unchanged in every fold. Two classifiers use 68 and 78 inputs; the union contains 78 unique features.')
    lines = ['# BlindSpot v0.4 measured results', '',
             'Same previously inspected 7,500-frame public sample. Compare methods within each protocol; do not compare one protocol score with another.', '',
             '| Group | Features | Whole-run spell F1 | Fixed-time spell F1 | Whole-run dead F1 | Fixed-time dead F1 |',
             '|---|---:|---:|---:|---:|---:|']
    if OUT.name == 'blindspot-v041':
        lines[0] = '# BlindSpot v0.4.1: team-only contacts results'
    elif OUT.name == 'blindspot-v05':
        lines[0] = '# BlindSpot v0.5: structural-prior verification'
    rows = ''
    for name, value in scores.items():
        w, f = value['whole_run'], value['fixed_time']
        numbers = [w['spell'], f['spell'], w['dead'], f['dead']]
        lines.append(f"| {name} | {value['n_features']} | " + ' | '.join(f'{v:.4f}' for v in numbers) + ' |')
        rows += '<tr><td>'+html.escape(name)+'</td><td>'+str(value['n_features'])+'</td>'+''.join(f'<td>{v:.4f}</td>' for v in numbers)+'</tr>'
    lines += ['', 'Provisional export group: **'+chosen+'**.', selection['rule'], '', selection['scope'], '',
              'The 300-frame embargo excludes training labels. New offline tracking features can share raw context across folds and use entire-half statistics. All baseline folds reproduce the frozen reference where available.', '',
              'Per-fold diagnostics, prediction CSVs, mask hashes, source/input hashes and environment are stored next to this report. Exported candidates fit all public labels; their public-window score is training fit.']
    if head_note:
        lines += ['', '## Team-only head isolation', '', head_note]
    training_note = ''
    training_path = OUT / 'training_fit_only/reports/CoreOrigin.json'
    if training_path.exists():
        fit = json.loads(training_path.read_text())
        note = (f"Direct public-window score of the all-public-trained candidate: spell F1 {fit['spell_f1_mean']:.4f}, "
                f"macro F1 {fit['macro_f1']:.4f}. This includes training labels and is NOT held-out performance or private-match accuracy.")
        lines += ['', '## Training fit only', '', note]
        training_note = '<section><h2>Training fit only</h2><p>'+html.escape(note)+'</p></section>'
    (OUT / 'RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    diagnostic_rows = ''
    for name in scores:
        d = diagnostics[name]['whole_run']
        diagnostic_rows += '<tr><td>'+html.escape(name)+'</td>'+''.join('<td>'+str(d[k])+'</td>' for k in
            ['wrong_live_team_frames', 'live_called_dead_frames', 'dead_called_live_frames'])+f"<td>{d['transitions']['25']['f1']:.4f}</td></tr>"
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CoreOrigin · v0.4 experiments</title><style>body{background:#0a1220;color:#e8eff8;font:16px system-ui;margin:0}main{max-width:1150px;margin:auto;padding:32px 22px}h1{font-size:42px}p{color:#b8c9df;line-height:1.6}section{background:#142238;border:1px solid #304763;padding:24px;margin:22px 0;border-radius:14px}a{color:#79e5c5}table{border-collapse:collapse;width:100%}th,td{padding:12px;text-align:left;border-bottom:1px solid #304763;font-variant-numeric:tabular-nums}.scroll{overflow:auto}.badge{color:#79e5c5}</style><main><div class="badge">COREORIGIN / BLINDSPOT · v0.4</div><h1>What changed the score?</h1><p>Seven fixed feature groups · two public development protocols · CPU training</p><p><a href="../index.html">All versions</a> · <a href="scores.json">Scores JSON</a> · <a href="RESULTS.md">Results document</a></p><section><h2>Score comparison</h2><p>Compare methods within each column. Both protocols reuse the same inspected five minutes. Group selection on this sample is exploratory and does not establish private-match rank.</p><div class="scroll"><table><tr><th>Group</th><th>Features</th><th>Whole-run spell F1</th><th>Fixed-time spell F1</th><th>Whole-run dead F1</th><th>Fixed-time dead F1</th></tr>'''+rows+'''</table></div></section><section><h2>Provisional export</h2><p>Chosen group: <strong>'''+html.escape(chosen)+'''</strong>. Require a gain greater than 0.06 in both spell-F1 protocols; otherwise retain baseline. This is a conservative heuristic, not a statistical significance threshold.</p><p>Candidate training uses all public labels. Its score on those labels is training fit, not validation. <a href="selection.json">Selection record</a></p></section><section><h2>Whole-run error breakdown</h2><p>Frame error counts pooled across the four evaluation folds. Transition F1 uses the official one-second tolerance and the same fold clipping for every group.</p><div class="scroll"><table><tr><th>Group</th><th>Wrong live team</th><th>Live called dead</th><th>Dead called live</th><th>Transition F1 @ 25 frames</th></tr>'''+diagnostic_rows+'''</table></div><p><a href="diagnostics.json">Both protocols and both timing tolerances</a></p></section><section><h2>Verification and scope</h2><p>Baseline gate checks the frozen v0.3.1 scores. Fold predictions, masks and source/input hashes are saved. New offline features can use raw context across a whole half; the 300-frame embargo excludes training labels, not every shared tracking observation.</p><p><a href="run_provenance.json">Provenance</a> · <a href="environment.json">Environment</a> · <a href="run.log">Execution log</a></p><p>Independent permitted validation remains necessary. No competition submission has been made.</p></section></main></html>'''
    page = page.replace('</main>', training_note+'</main>')
    page = page.replace('Seven fixed feature groups', str(len(scores))+' fixed approaches')
    if OUT.name == 'blindspot-v041':
        page = page.replace('v0.4', 'v0.4.1')
    elif OUT.name == 'blindspot-v05':
        page = page.replace('v0.4', 'v0.5')
    if head_note:
        page = page.replace('</main>', '<section><h2>Head isolation</h2><p>'+html.escape(head_note)+'</p><p><a href="head_invariants.json">Per-fold checks</a></p></section></main>')
    (OUT / 'index.html').write_text(page, encoding='utf-8')
    print(json.dumps(selection, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, default=OUT)
    main(parser.parse_args().experiment)
