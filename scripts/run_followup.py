"""Resume-safe v0.2 audits, grouped evaluation, query benchmark and review lab."""

import argparse
import hashlib
import json
import platform
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from possession.data import extract, sha256
from possession.model import predict, choose_queries
from possession.stoppage import scores as stoppage_scores
from possession.validation import grouped_split, query_benchmark, class_diagnostics
from possession.calibrate import fit, scores as calibrated_scores
from possession.quality import scan_source
from possession.review import build_review
from scripts.run_experiment import save_json, save_submission, summary
from evaluation import possession_metrics as pm


class Checkpoints:
    """Reuse only stages with matching run fingerprint AND every artifact hash."""

    def __init__(self, out, fingerprint, resume):
        self.out, self.fingerprint, self.resume = Path(out), fingerprint, resume
        self.status = []

    def run(self, name, action):
        directory = self.out / name
        marker = directory / 'checkpoint.json'
        if self.resume and marker.exists():
            try:
                saved = json.loads(marker.read_text())
            except (ValueError, OSError):
                saved = {}
            if (saved.get('fingerprint') == self.fingerprint and saved.get('artifacts') and
                    all((directory / path).is_file() and sha256(directory / path) == digest
                        for path, digest in saved['artifacts'].items())):
                print('Verified checkpoint; reused:', name, flush=True)
                self.status.append({'stage': name, 'status': 'reused'})
                return
        directory.mkdir(parents=True, exist_ok=True)
        if marker.exists():
            marker.unlink()
        started = time.monotonic()
        action(directory)
        artifacts = {path.relative_to(directory).as_posix(): sha256(path)
                     for path in sorted(directory.rglob('*')) if path.is_file() and path != marker and '.partial' not in path.name}
        temporary = marker.with_suffix('.partial.json')
        save_json(temporary, {'fingerprint': self.fingerprint, 'artifacts': artifacts})
        temporary.replace(marker)
        seconds = round(time.monotonic() - started, 3)
        self.status.append({'stage': name, 'status': 'completed', 'seconds': seconds})
        print('Checkpoint saved:', name, seconds, 'seconds', flush=True)


def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / 'configs/evidence_v1.json').read_text())
    arrays, meta = extract(args.data_dir, out.parent / 'cache')
    gt_path = ROOT / 'evaluation/gt/BAR-ATH_possession_gt_public.csv'
    code = sorted((ROOT / 'possession').glob('*.py')) + [Path(__file__), ROOT / 'games.py', ROOT / 'evaluation/possession_metrics.py']
    source = {p.relative_to(ROOT).as_posix(): sha256(p) for p in code}
    baselines = sorted((ROOT / 'evaluation/results').glob('baseline_*.csv'))
    provenance = {'source': source, 'inputs': meta['input_sha256'], 'config': config,
                  'public_truth_sha256': sha256(gt_path),
                  'baselines': {p.name: sha256(p) for p in baselines},
                  'options': {'packets': args.packets, 'review_video': args.review_video, 'review_budget': args.review_budget}}
    # Include optional inputs, so a different/missing video or broadcast invalidates checkpoints.
    provenance['optional_assets'] = {p.relative_to(Path(args.data_dir)).as_posix():
                                     ({'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns} if p.suffix == '.mp4' else {'sha256': sha256(p)})
                                     for source_name in ('broadcast', 'tactical') for p in (Path(args.data_dir) / source_name).glob('*')
                                     if p.is_file() and (source_name == 'broadcast' or p.suffix == '.mp4')}
    fingerprint = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    stages = Checkpoints(out, fingerprint, args.resume)

    def predictions(directory):
        base, posterior, histories, diagnostics = predict(arrays, config)
        candidate, _, _, _ = predict(arrays, config, samples=0, score_override=stoppage_scores(arrays, config))
        without_tags, _, _, _ = predict(arrays, config, use_tags=False, samples=0)
        disagreement = (without_tags != base).astype(np.float32)
        save_submission(directory / 'evidence_v1.csv', base)
        save_submission(directory / 'stoppage_v2.csv', candidate)
        np.savez_compressed(directory / 'inference.npz', posterior=posterior, histories=histories, disagreement=disagreement)
        save_json(directory / 'fixed_candidate.json', {'kind': 'bounded_multisecond_inactivity_outside_stillness',
                  'max_logit_correction': 4., 'status': 'development_candidate_only',
                  'main_choice': 'evidence_v1; no automatic promotion based on public scores'})
    stages.run('predictions', predictions)
    base = pd.read_csv(out / 'predictions/evidence_v1.csv')['label'].to_numpy()
    candidate = pd.read_csv(out / 'predictions/stoppage_v2.csv')['label'].to_numpy()
    with np.load(out / 'predictions/inference.npz') as z:
        posterior, disagreement = z['posterior'], z['disagreement']

    def quality(directory):
        result = {}
        for source_name in ('tactical', 'broadcast'):
            if not (Path(args.data_dir) / source_name / f'BAR-ATH_{source_name}_atd.txt').exists():
                result[source_name] = {'status': 'not_available_in_this_runtime'}
                continue
            result[source_name] = scan_source(args.data_dir, source_name, args.packets)
            print('Audited', source_name, flush=True)
        save_json(directory / 'audit.json', result)
    stages.run('quality', quality)

    # Label-free predictions and query features exist before opening label values.
    truth = pm.read_label_csv(gt_path)['label'].to_numpy()

    def validation(directory):
        train, test, split = grouped_split(truth)
        indices = np.flatnonzero(train)
        learned = fit(arrays, config, truth, int(indices[0]), int(indices[-1] + 1), train_mask=train)
        learned['split'] = split
        learned['public_truth_sha256'] = sha256(gt_path)
        save_json(directory / 'calibration_model.json', learned)
        calibrated, _, _, _ = predict(arrays, config, samples=0, score_override=calibrated_scores(arrays, config, learned))
        save_submission(directory / 'grouped_calibrated.csv', calibrated)
        masked = np.where(test, truth, 'X')
        np.savez_compressed(directory / 'split_masks.npz', train=train, test=test)
        save_json(directory / 'split.json', split)
        predictions = {'evidence_v1': base, 'stoppage_v2': candidate, 'grouped_calibrated': calibrated}
        predictions.update({p.stem: pd.read_csv(p)['label'].to_numpy() for p in baselines})
        scores, diagnostics = {}, {}
        for name, labels in predictions.items():
            scores[name] = summary(masked, labels, 'Whole-run public development holdout; previously examined same-match data')
            diagnostics[name] = class_diagnostics(masked, labels)
        full = {name: summary(truth, labels, 'Five-minute public development sample; label-free fixed method')
                for name, labels in predictions.items() if name != 'grouped_calibrated'}
        save_json(directory / 'scores.json', {'grouped_development': scores, 'full_public': full})
        save_json(directory / 'error_analysis.json', diagnostics)
        for name, score in scores.items():
            print(f"Grouped development {name}: spellF1 {score['spell_f1_mean']:.4f}, macroF1 {score['macro_f1']:.4f}", flush=True)
    stages.run('validation', validation)

    def query_eval(directory):
        # Fixed contiguous public window; no ground-truth boundaries guide selection.
        benchmark = query_benchmark(posterior, disagreement, arrays, config, truth, base, [(1, 11185, 18684)])
        save_json(directory / 'query_benchmark.json', benchmark)
        save_json(directory / 'review_plan.json', choose_queries(posterior, disagreement, arrays, config))
    stages.run('query_evaluation', query_eval)

    def review(directory):
        queries = json.loads((out / 'query_evaluation/review_plan.json').read_text())
        result = build_review(args.data_dir, directory, queries, base, posterior, encode=args.review_video, budget=args.review_budget)
        save_json(directory / 'summary.json', result)
    stages.run('review', review)

    save_json(out / 'manifest.json', {'version': '0.2.0', 'fingerprint': fingerprint, 'provenance': provenance,
                                     'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__},
                                     'stages': stages.status, 'main_prediction': 'predictions/evidence_v1.csv',
                                     'candidate_policy': 'No automatic promotion; no private truth or manual prediction corrections',
                                     'review': 'review/index.html'})
    write_dashboard(out)
    if args.archive:
        archive = shutil.make_archive(str(out.parent / 'BlindSpot_v02_results'), 'zip', out)
        print('Private result archive:', archive, flush=True)
    print('Verified follow-up outputs:', out.resolve(), flush=True)


def write_dashboard(out):
    import html
    scores = json.loads((out / 'validation/scores.json').read_text())
    benchmark = json.loads((out / 'query_evaluation/query_benchmark.json').read_text())
    split = json.loads((out / 'validation/split.json').read_text())
    rows = ''.join(f"<tr><td>{html.escape(name)}</td><td>{r['spell_f1_mean']:.4f}</td><td>{r['macro_f1']:.4f}</td><td>{r['per_class']['D']['f1']:.4f}</td></tr>" for name, r in scores['grouped_development'].items())
    queries = ''.join(f"<tr><td>{name}</td><td>{r['review_seconds']:.2f}</td><td>{r['error_rate']:.3f}</td><td>{r['fraction_of_all_errors_covered']:.3f}</td><td>{r['true_boundaries_in_clips']}/{r['true_boundaries_total']}</td></tr>" for name, r in benchmark['policies'].items())
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>BlindSpot v0.2 experiments</title><style>body{{background:#0a1220;color:#e6edf7;font:16px system-ui}}main{{max-width:1150px;margin:auto;padding:30px}}section{{background:#142238;padding:24px;border-radius:14px;margin:20px 0}}h1{{font-size:40px}}td,th{{padding:12px;border-bottom:1px solid #36506b;text-align:left}}table{{width:100%;border-collapse:collapse}}p{{color:#b4c8dd}}a{{color:#73dfc0}}pre{{white-space:pre-wrap}}</style><main><h1>BlindSpot · experiment lab</h1><p>v0.2 · reproducible diagnostics, honest development comparisons, synchronized evidence review.</p><section><h2><a href="review/index.html">Open synchronized review lab →</a></h2><p>Selected tactical video clips, pitch snapshots, model states and portable review notes. Review notes never overwrite predictions.</p></section><section><h2>Whole-possession development split</h2><p>Complete H/A/D runs remain together. {split['train_frames']} training frames · {split['actual_gap_frames']} embargo frames · {split['test_frames']} evaluation frames. This public footage has already been examined; it is not an independent final test.</p><table><tr><th>Method</th><th>Spell F1</th><th>Macro F1</th><th>Dead-ball F1</th></tr>{rows}</table></section><section><h2>Does query selection find mistakes?</h2><p>After label-independent selection, public labels measure error coverage in fixed two-second clips. This does not measure correction, improved predictions or reviewer time saved. Whole-window error rate: {benchmark['overall_error_rate']:.3f}.</p><table><tr><th>Policy</th><th>Review seconds</th><th>Error rate</th><th>Error coverage</th><th>Boundaries covered</th></tr>{queries}</table><p>Random controls: 30 deterministic seeds. Mean error rate {benchmark['random_30_seeds']['mean_error_rate']:.3f}; 5–95 percentile range {benchmark['random_30_seeds']['p05_p95_error_rate']}.</p></section><section><h2>Inspect the evidence</h2><p><a href="quality/audit.json">Tracking and video audit</a> · <a href="validation/error_analysis.json">Stoppage error analysis</a> · <a href="validation/split.json">Exact evaluation groups</a> · <a href="manifest.json">Run provenance</a></p><p>The original label-free prediction remains the main output. Candidate scores cannot establish hidden-match performance. No full pixel integrity scan or independent human review has been completed.</p></section></main></html>'''
    (out / 'index.html').write_text(page, encoding='utf-8')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', default=str(ROOT / 'data'))
    p.add_argument('--out', default=str(ROOT / 'data/experiments/blindspot-v2'))
    p.add_argument('--resume', action='store_true')
    p.add_argument('--packets', action='store_true', help='Scan compressed video packets; not full decode')
    p.add_argument('--review-video', action='store_true', help='Encode and verify local private video snippets')
    p.add_argument('--review-budget', type=int, default=12)
    p.add_argument('--archive', action='store_true')
    args = p.parse_args()
    if not 0 <= args.review_budget <= 12:
        p.error('review budget must be between 0 and 12')
    run(args)
