"""Fixed v0.3 experiments: SI units, factorized motion and purged cross-fitting."""

import argparse
import hashlib
import html
import json
from pathlib import Path
import platform
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import sklearn

from possession.data import extract, sha256
from possession.metric_features import extract_metric, design
from possession.model import predict, choose_queries
from possession.champion import (fit_motion, motion_probability, decode_factorized, purged_folds,
                                 fit_classifier, classifier_probability, active_mask)
from possession.report import write_report
from scripts.run_experiment import save_json, save_submission
from scripts.run_followup import Checkpoints
from evaluation import possession_metrics as pm


def evaluate(truth, labels, mask, bootstrap=0):
    ids = np.flatnonzero(mask)
    if not len(ids) or (np.diff(ids) != 1).any():
        raise ValueError('Metric subsets must be a nonempty contiguous window')
    return pm.evaluate(pm.label_codes(truth[ids]), pm.label_codes(labels[ids]), bootstrap_n=bootstrap)


def pooled(folds):
    confusion = np.sum([r['confusion'] for r in folds], axis=0)
    scored = pm.scored_confusion(confusion)
    counts = {}
    f1s = []
    for threshold in pm.SPELL_IOUS:
        matched = sum(r['spell_f1'][threshold]['n_matched'] for r in folds)
        spurious = sum(r['spell_f1'][threshold]['n_spurious'] for r in folds)
        missed = sum(r['spell_f1'][threshold]['n_missed'] for r in folds)
        f1 = 2*matched/max(1, 2*matched+spurious+missed)
        counts[str(threshold)] = {'n_matched': matched, 'n_spurious': spurious, 'n_missed': missed, 'f1': f1}
        f1s.append(f1)
    return {'macro_f1': pm.macro_f1(scored), 'spell_f1_mean': float(np.mean(f1s)),
            'dead_f1': pm.per_class_prf(scored, 2)[2], 'n_graded': int(scored.sum()),
            'n_gt_spells': sum(r['n_gt_spells'] for r in folds),
            'n_pred_spells': sum(r['n_pred_spells'] for r in folds),
            'confusion': confusion.tolist(), 'spell_f1': counts,
            'scope': 'Pooled counts across four complete-run public development folds; boundary clipping applies equally to all methods. No stitched prediction scored.'}


def run(args):
    started = time.monotonic()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(args.config).read_text())
    arrays, metric_meta = extract_metric(args.data_dir, out.parent / 'cache', config)
    legacy, _ = extract(args.data_dir, out.parent / 'cache')
    matrix, names = design(arrays, config)
    source_files = sorted((ROOT / 'possession').glob('*.py')) + [Path(__file__), ROOT / 'scripts/run_followup.py',
                     ROOT / 'evaluation/possession_metrics.py', ROOT / 'games.py']
    provenance = {'source_sha256': {p.relative_to(ROOT).as_posix(): sha256(p) for p in source_files},
                  'input_sha256': metric_meta['input_sha256'], 'feature_fingerprint': metric_meta['fingerprint'], 'config': config}
    gt_path = ROOT / 'evaluation/gt/BAR-ATH_possession_gt_public.csv'
    provenance['public_truth_sha256'] = sha256(gt_path)
    baselines = sorted((ROOT / 'evaluation/results').glob('baseline_*.csv'))
    provenance['baseline_sha256'] = {p.name: sha256(p) for p in baselines}
    fingerprint = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    save_json(out / 'fixed_protocol.json', {'config': config, 'fingerprint': fingerprint,
              'sequence': 'Generate label-free predictions before reading reference label values. Three fixed methods, no grid search.',
              'release_policy': 'Development candidates only. Historical output retained; no automatic promotion based solely on public scores.'})
    stages = Checkpoints(out, fingerprint, args.resume)

    def label_free(directory):
        # Historical reference retains raw speed handling; metadata is inconsistent.
        old, _, _, _ = predict(legacy, config, samples=0)
        fixed, _, _, _ = predict(arrays, config, samples=0)
        motion = fit_motion(matrix, names, config)
        p_dead = motion_probability(matrix, motion)
        final, posterior, histories = decode_factorized(arrays, config, p_dead)
        save_submission(directory / 'legacy_v1.csv', old)
        save_submission(directory / 'trajectory_v1.csv', fixed)
        save_submission(directory / 'motion_factorized.csv', final)
        joblib.dump(motion, directory / 'motion_model.joblib')
        save_json(directory / 'motion_model.json', motion['audit'])
        np.savez_compressed(directory / 'inference.npz', posterior=posterior, histories=histories, dead_probability=p_dead)
    stages.run('label_free', label_free)
    predictions = {name: pd.read_csv(out / f'label_free/{name}.csv')['label'].to_numpy()
                   for name in ('legacy_v1', 'trajectory_v1', 'motion_factorized')}
    predictions.update({p.stem: pd.read_csv(p)['label'].to_numpy() for p in baselines})
    # No reference-label values are available to the label-free generation above.
    truth = pm.read_label_csv(gt_path)['label'].to_numpy()
    valid = truth != 'X'
    folds = purged_folds(truth, config['development_folds'], config['embargo_frames'], config['feature_context_radius_frames'])

    def supervised(directory):
        all_scores = {name: [] for name in predictions}
        all_scores['cross_fitted_classifier'] = []
        audits = []
        for train, test, info in folds:
            model, audit = fit_classifier(matrix, names, truth, train, config)
            probability = classifier_probability(model, matrix, names).clip(1e-6, 1.)
            decode_config = dict(config, history_samples=0)
            labels, _, _ = decode_factorized(arrays, decode_config, probability[:, 2],
                                             np.log(probability[:, :2] / np.maximum(1e-6, probability[:, :2].sum(1))[:, None]))
            save_submission(directory / f"fold_{info['fold']}_predictions.csv", labels)
            joblib.dump(model, directory / f"fold_{info['fold']}_model.joblib")
            for name, prediction in {**predictions, 'cross_fitted_classifier': labels}.items():
                all_scores[name].append(evaluate(truth, prediction, test))
            audit.update(info)
            audit['training_class_frames'] = {state: int(((truth == state) & train).sum()) for state in ('H', 'A', 'D')}
            audits.append(audit)
            print(f"Completed purged fold {info['fold']}: train {info['train_frames']}, test {info['test_frames']}", flush=True)
        model, audit = fit_classifier(matrix, names, truth, valid, config)
        probability = classifier_probability(model, matrix, names).clip(1e-6, 1.)
        labels, posterior, histories = decode_factorized(arrays, config, probability[:, 2],
                            np.log(probability[:, :2] / np.maximum(1e-6, probability[:, :2].sum(1))[:, None]))
        joblib.dump(model, directory / 'full_public_model.joblib')
        save_submission(directory / 'full_public_trained_candidate.csv', labels)
        np.savez_compressed(directory / 'full_public_inference.npz', posterior=posterior, histories=histories)
        save_json(directory / 'full_public_training.json', audit)
        save_json(directory / 'fold_audits.json', audits)
        save_json(directory / 'fold_scores.json', all_scores)
        save_json(directory / 'pooled_scores.json', {name: pooled(scores) for name, scores in all_scores.items()})
    stages.run('supervised', supervised)

    def evaluation(directory):
        scores = {name: evaluate(truth, labels, valid, bootstrap=300) for name, labels in predictions.items()}
        trained = pd.read_csv(out / 'supervised/full_public_trained_candidate.csv')['label'].to_numpy()
        save_json(directory / 'full_public_label_free_scores.json', scores)
        save_json(directory / 'full_fit_descriptive_only.json', {
            'warning': 'This model fits ALL public labels. Public score includes training and is NOT validation.',
            'scores': evaluate(truth, trained, valid)})
        # Aggregates only; individual tracking extracts remain in ignored data/.
        diagnostics = {}
        for state in ('H', 'A', 'D'):
            mask = truth == state
            diagnostics[state] = {'frames': int(mask.sum()),
                'median_reported_speed_raw': float(np.nanmedian(legacy['median_speed'][mask])),
                'median_trajectory_speed_mps': float(np.nanmedian(arrays['median_speed'][mask])),
                'mean_stationary_fraction_mps': float(np.nanmean(arrays['stationary'][mask])),
                'observed_ball_fraction': float(np.isfinite(arrays['ball'][mask]).all(1).mean())}
        save_json(directory / 'unit_diagnostics.json', diagnostics)
        save_json(directory / 'speed_unit_audit.json', metric_meta['unit_audit'])
        for name, score in scores.items():
            print(f"Full public {name}: spellF1 {score['spell_f1_mean']:.4f}, deadF1 {score['per_class']['D']['f1']:.4f}", flush=True)
        for name, score in json.loads((out / 'supervised/pooled_scores.json').read_text()).items():
            print(f"Four-fold development {name}: pooled spellF1 {score['spell_f1_mean']:.4f}, deadF1 {score['dead_f1']:.4f}", flush=True)
    stages.run('evaluation', evaluation)

    def robustness(directory):
        model = joblib.load(out / 'label_free/motion_model.joblib')  # own just-generated artifact only
        altered = dict(arrays, ball=arrays['ball'].copy(), nearest=arrays['nearest'].copy())
        affected = np.zeros(len(truth), dtype=bool)
        rng = np.random.default_rng(config['seed'])
        from games import half_video_blocks, GAME
        windows = []
        for half, first, last in half_video_blocks(GAME):
            candidates = np.arange(first + 150, last - 225, 250)
            rng.shuffle(candidates)
            for frame in candidates[:10]:
                stop = int(frame) + 75
                affected[frame:stop] = True
                windows.append({'first_frame': int(frame), 'stop_frame': stop, 'half': half})
        altered['ball'][affected] = np.nan
        altered['nearest'][affected] = np.nan
        perturbed_matrix, _ = design(altered, config)
        p_dead = motion_probability(perturbed_matrix, model)
        labels, _, _ = decode_factorized(altered, dict(config, history_samples=0), p_dead)
        save_submission(directory / 'artificial_gap_predictions.csv', labels)
        full = active_mask(len(truth))
        save_json(directory / 'artificial_gap_stress.json', {
            'selection': 'Seeded three-second gaps selected without labels, ten in each half; fitted motion model frozen.',
            'windows': windows, 'n_removed_ball_frames': int((affected & np.isfinite(arrays['ball']).all(1)).sum()),
            'changed_fraction_inside_mask': float((labels[affected] != predictions['motion_factorized'][affected]).mean()),
            'changed_fraction_over_halves': float((labels[full] != predictions['motion_factorized'][full]).mean()),
            'graded_public_affected_frames': int((affected & valid).sum()),
            'warning': 'Artificial occlusion sensitivity, not accuracy on natural gaps or new match validation.'})
    stages.run('robustness', robustness)

    candidate = pd.read_csv(out / 'supervised/full_public_trained_candidate.csv')['label'].to_numpy()
    with np.load(out / 'supervised/full_public_inference.npz') as z:
        posterior, histories = z['posterior'], z['histories']
    votes = np.stack([pd.read_csv(out / f'supervised/fold_{i}_predictions.csv')['label'].to_numpy()
                      for i in range(config['development_folds'])])
    disagreement = (votes != candidate).mean(0)
    queries = choose_queries(posterior, disagreement, arrays, config, decoded_labels=candidate)
    save_json(out / 'review_queries.json', queries)

    def candidate_bundle(directory):
        candidate_hash = save_submission(directory / 'CoreOrigin.csv', candidate)
        # Covariate-envelope flags, not calibrated out-of-distribution probabilities.
        low, high = np.quantile(matrix[valid], [.01, .99], axis=0)
        envelope = ((matrix < low) | (matrix > high)).mean(1)
        from games import half_video_blocks, GAME, GAMES, anchor_video_frame
        coverage = {}
        for half, first, last in half_video_blocks(GAME):
            ids = np.arange(first, last+1)
            anchor_name = 'kickoff' if half == 1 else 'kickoff_2nd'
            before = max(first, min(last+1, anchor_video_frame(GAMES[GAME][anchor_name])-config['kickoff_prior_guard_frames']))
            coverage[str(half)] = {'frames': len(ids), 'label_counts': {state: int((candidate[ids] == state).sum()) for state in ('H','A','D')},
                'missing_ball_fraction': float((~np.isfinite(arrays['ball'][ids]).all(1)).mean()),
                'fraction_with_any_fold_model_disagreement': float((disagreement[ids] > 0).mean()),
                'mean_feature_envelope_exceedance_fraction': float(envelope[ids].mean()),
                'metadata_pre_kickoff_frames': before-first,
                'dead_fraction_in_metadata_pre_kickoff': float((candidate[first:before] == 'D').mean()) if before>first else None}
        save_json(directory / 'coverage_audit.json', {'halves': coverage,
            'interpretation': 'Agreement among related models and training-envelope flags are review signals, not correctness certificates or calibrated OOD scores.'})
        stability = np.column_stack([np.arange(len(candidate)), disagreement, envelope])
        pd.DataFrame(stability, columns=['frame','fold_model_disagreement','feature_envelope_exceedance']).assign(frame=np.arange(len(candidate))).to_csv(directory / 'review_risk.csv', index=False, lineterminator='\n')
        all_fold_scores = json.loads((out / 'supervised/fold_scores.json').read_text())
        # JSON converts threshold keys to strings; restore for the pooled helper.
        for results in all_fold_scores.values():
            for item in results:
                item['spell_f1'] = {float(k): v for k,v in item['spell_f1'].items()}
        stability_scores = []
        for omitted in range(4):
            method = pooled([r for i,r in enumerate(all_fold_scores['cross_fitted_classifier']) if i != omitted])
            reference = pooled([r for i,r in enumerate(all_fold_scores['baseline_smooth125']) if i != omitted])
            stability_scores.append({'omitted_fold': omitted, 'classifier_spell_f1': method['spell_f1_mean'],
                                     'reference_spell_f1': reference['spell_f1_mean'],
                                     'difference': method['spell_f1_mean']-reference['spell_f1_mean']})
        save_json(directory / 'fold_sensitivity.json', {'leave_one_evaluation_block_out': stability_scores,
            'scope': 'Descriptive sensitivity of pooled counts, without retraining or model selection; only four correlated development folds.'})
        save_json(directory / 'candidate_manifest.json', {'status': 'provisional_development_candidate',
            'csv': 'CoreOrigin.csv', 'sha256': candidate_hash, 'fingerprint': fingerprint,
            'model': 'Symmetric classifier trained on all permitted public labels, with factorized sequence decoding',
            'training_warning': 'All public labels used for this exported model. Its full-public score is not validation. Four-fold results come from separate models.',
            'metadata_prior': 'Dead before published kickoff anchors, stopping 25 frames short; no private truth used.',
            'team_name': 'CoreOrigin',
            'public_code_repository': 'https://github.com/7dracoder/more-than-a-hack-2027',
            'remaining': ['Independent validation beyond the public example', 'Verify cloud reproduction', 'Confirm final release selection', 'Publish reviewed source snapshot to team repository', 'Submit through organizer form only when instructed']})
    stages.run('candidate', candidate_bundle)
    hashes = {p.relative_to(out).as_posix(): sha256(p) for folder in ('label_free', 'supervised', 'robustness', 'candidate')
              for p in sorted((out / folder).glob('*.csv'))}
    manifest = {'version': config.get('version', '0.3.0'), 'fingerprint': fingerprint, 'provenance': provenance,
                'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
                                'scikit_learn': sklearn.__version__},
                'output_csv_sha256': hashes, 'stages': stages.status, 'elapsed_seconds': round(time.monotonic()-started, 3),
                'scope': 'Fixed development experiments on previously examined public footage; no hidden-match result.',
                'prediction_choice': 'Provisional supervised development candidate, not a final competition submission. Historical CSVs retained.',
                'posterior_scope': ('Live/dead marginals with owner support conditional on the MAP live/dead segmentation; sampled histories use their own conditional owner inference.' if config.get('reset_owner_at_stoppage') else 'Independent live/dead and owner marginals.')}
    save_json(out / 'manifest.json', manifest)
    full_scores = json.loads((out / 'evaluation/full_public_label_free_scores.json').read_text())
    report_scores = {name: dict(value, scope='Full public; no label fitting') for name, value in full_scores.items()}
    cross_fitted = json.loads((out / 'supervised/pooled_scores.json').read_text())
    report_scores.update({name+'_four_fold_development': value for name,value in cross_fitted.items()})
    write_report(out / 'evidence_report.html', candidate, posterior, histories,
                 disagreement, arrays, queries, report_scores, manifest,
                 notice='Displayed timeline: provisional classifier fitted on all public labels. Cross-fitted scores use separate held-out models. Fold disagreement and model marginals are review signals, not calibrated confidence.')
    write_dashboard(out)
    if args.archive:
        archive_name = 'BlindSpot_v03_results' if out.name == 'blindspot-v3' else out.name + '_results'
        print('Private archive:', shutil.make_archive(str(out.parent / archive_name), 'zip', out), flush=True)
    print('Candidate suite complete:', out.resolve(), flush=True)


def write_dashboard(out):
    pooled_scores = json.loads((out / 'supervised/pooled_scores.json').read_text())
    public = json.loads((out / 'evaluation/full_public_label_free_scores.json').read_text())
    def rows(values, full=False):
        return ''.join(f"<tr><td>{html.escape(name)}</td><td>{r['spell_f1_mean']:.4f}</td><td>{r['macro_f1']:.4f}</td><td>{(r['per_class']['D']['f1'] if full else r['dead_f1']):.4f}</td><td>{r['n_pred_spells']}</td></tr>" for name,r in values.items())
    table = '<tr><th>Method</th><th>Spell F1</th><th>Macro F1</th><th>Dead F1</th><th>Possessions</th></tr>'
    text = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>BlindSpot v0.3 — measured development</title><style>body{{background:#091221;color:#e5eef9;font:16px system-ui}}main{{max-width:1180px;margin:auto;padding:30px}}h1{{font-size:42px}}section{{background:#142239;border:1px solid #304460;border-radius:15px;padding:25px;margin:22px 0}}td,th{{padding:12px;text-align:left;border-bottom:1px solid #304460}}table{{width:100%;border-collapse:collapse}}p{{color:#b7c9df}}a{{color:#7ce9c9}}.badge{{color:#7ce9c9}}</style><main><span class="badge">CoreOrigin · BlindSpot v0.3 research laboratory</span><h1>Audit the measurements.<br>Test the possession model.</h1><p>Trajectory-based SI units · separated live/dead inference · symmetric learning · purged cross-fitting</p><section><h2>What changed</h2><p>The source metadata says km/h, but displacement checks show that the supplied speed values empirically resemble m/s. Blindly dividing by 3.6 is not a proven correction. This version derives movement from normalized positions, pitch dimensions and frame timing, rejecting short windows containing gaps or impossible jumps. The earlier metadata-scaling experiment is retained as a pilot with its interpretation corrected.</p><p>The motion model learns two movement regimes from unlabelled match inputs. It separates live/dead inference from the uncertain choice of team. The small CPU classifier enforces home/away symmetry and is tested with whole-run folds and a 12-second training embargo.</p></section><section><h2>Four-fold public development comparison</h2><p>Pooled match counts across the same four contiguous evaluation blocks. Entire labelled runs stay together. Each learned prediction comes from a model that excludes that block and nearby runs. No stitched cross-fold prediction is scored; no model is clamped to known boundary labels. This previously inspected five-minute sample is not independent-match validation.</p><table>{table}{rows(pooled_scores)}</table></section><section><h2>Full public sample: label-free methods</h2><p>These methods use no possession labels for fitting. The baseline that wins this sample is not the one that wins the full match. New public gains cannot establish private-match ranking.</p><table>{table}{rows(public, True)}</table></section><section><h2>Inspect and reproduce</h2><p><a href="evidence_report.html">Explore motion evidence and alternative histories</a> · <a href="evaluation/speed_unit_audit.json">Speed-unit consistency audit</a> · <a href="supervised/fold_audits.json">Fold provenance</a> · <a href="robustness/artificial_gap_stress.json">Missing-ball stress test</a> · <a href="manifest.json">Hashes and configuration</a></p><p>The separately exported full-public-trained candidate uses all public labels for fitting; its public score is descriptive training performance only. Restricted outputs remain local. Archive-relative links work in the extracted result folder, rather than Colab's inline preview.</p></section></main></html>'''
    version = json.loads((out / 'manifest.json').read_text())['version']
    text = text.replace('v0.3', 'v' + version)
    text = text.replace('The baseline that wins this sample is not the one that wins the full match.', 'The public-sample winner may differ from the full-match winner.')
    if (out / 'review/index.html').exists():
        text = text.replace('<a href="evidence_report.html">', '<a href="review/index.html">Open synchronized video review</a> · <a href="evidence_report.html">')
    (out / 'index.html').write_text(text, encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default=str(ROOT / 'data'))
    parser.add_argument('--out', default=str(ROOT / 'data/experiments/blindspot-v3'))
    parser.add_argument('--config', default=str(ROOT / 'configs/champion_v3.json'))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--archive', action='store_true')
    run(parser.parse_args())
