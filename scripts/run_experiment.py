"""Run BlindSpot inference, diagnostics, optional public calibration, and ablations."""

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from possession import __version__  # noqa: E402
from possession.data import extract, audit, sha256, intervals  # noqa: E402
from possession.model import predict, choose_queries, boundary_receipts  # noqa: E402
from possession.calibrate import fit, scores as calibrated_scores  # noqa: E402
from possession.report import write_report  # noqa: E402
from evaluation import possession_metrics as pm  # noqa: E402


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_submission(path, labels):
    table = pd.DataFrame({"frame": np.arange(len(labels)), "label": labels})
    reason = pm.validate_submission(table, 199319)
    if reason:
        raise ValueError(reason)
    table.to_csv(path, index=False, lineterminator="\n")
    return sha256(path)


def summary(gt, labels, scope):
    result = pm.evaluate(pm.label_codes(gt), pm.label_codes(labels))
    return {"macro_f1": result["macro_f1"], "spell_f1_mean": result["spell_f1_mean"],
            "n_pred_spells": result["n_pred_spells"], "n_gt_spells": result["n_gt_spells"],
            "n_graded": result["n_graded"], "per_class": result["per_class"], "scope": scope}


def run(args):
    started = time.monotonic()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.query_budget is not None:
        config["query_budget"] = args.query_budget
    if config["query_budget"] < 0 or config["step"] < 1 or config["history_samples"] < 0:
        raise ValueError("Invalid budget, step, or history count")
    arrays, feature_meta = extract(args.data_dir, out.parent / "cache", args.rebuild_cache)
    save_json(out / "data_audit.json", audit(arrays))
    print("Aligned features ready; decoding independent halves...", flush=True)
    base, posterior, histories, diagnostics = predict(arrays, config)
    predictions = {"evidence_v1": base}
    hashes = {"evidence_v1.csv": save_submission(out / "evidence_v1.csv", base)}
    # These fixed source/transition perturbations are not a parameter search.
    specs = [("without_tags", {"use_tags": False}), ("without_camera", {"use_camera": False}),
             ("lower_switch_cost", {"switch_scale": .75}), ("higher_switch_cost", {"switch_scale": 1.25})]
    variants = []
    if args.ablations:
        for name, kwargs in specs:
            labels, _, _, _ = predict(arrays, config, samples=0, **kwargs)
            predictions[name] = labels
            variants.append(labels)
            hashes[name + ".csv"] = save_submission(out / (name + ".csv"), labels)
            print("Ablation complete:", name, flush=True)
    disagreement = np.mean(np.array(variants) != base, axis=0) if variants else np.zeros(len(base))
    base_posterior, base_histories, base_diagnostics = posterior.copy(), histories.copy(), diagnostics
    base_disagreement = disagreement.copy()
    evaluation = {}
    # Evaluation labels are read only after label-free predictions have been produced.
    gt_path = ROOT / "evaluation/gt/BAR-ATH_possession_gt_public.csv"
    gt = pm.read_label_csv(gt_path)["label"].to_numpy()
    for name, labels in predictions.items():
        evaluation[name] = summary(gt, labels, "public 5 minutes, no public-label fitting")
    for path in sorted((ROOT / "evaluation/results").glob("baseline_*.csv")):
        labels = pm.read_label_csv(path)["label"].to_numpy()
        evaluation[path.stem] = summary(gt, labels, "public 5 minutes, organizer reference")

    final = base
    if args.train_public:
        train_start, train_stop, holdout_start, holdout_stop = 11185, 14185, 14435, 18685
        learned = fit(arrays, config, gt, train_start, train_stop)
        learned["public_truth_sha256"] = sha256(gt_path)
        learned["holdout_first_frame"] = holdout_start
        learned["holdout_stop_exclusive"] = holdout_stop
        learned["buffer_frames"] = holdout_start - train_stop
        save_json(out / "calibration_model.json", learned)
        score = calibrated_scores(arrays, config, learned)
        final, posterior, histories, diagnostics = predict(arrays, config, score_override=score)
        predictions["calibrated_v1"] = final
        hashes["calibrated_v1.csv"] = save_submission(out / "calibrated_v1.csv", final)
        holdout = np.full(len(gt), "X", dtype="<U1")
        holdout[holdout_start:holdout_stop] = gt[holdout_start:holdout_stop]
        evaluation["calibrated_public_includes_training"] = summary(gt, final, "DESCRIPTIVE ONLY: includes training frames")
        evaluation["calibrated_blocked_holdout"] = summary(holdout, final, "170-second same-match held-out block; 10-second buffer")
        evaluation["evidence_blocked_holdout"] = summary(holdout, base, "same 170-second block; no label fitting")
        for path in sorted((ROOT / "evaluation/results").glob("baseline_*.csv")):
            evaluation[path.stem + "_blocked_holdout"] = summary(holdout, pm.read_label_csv(path)["label"].to_numpy(), "same 170-second block")
        # Source-sensitivity models use the same fixed training block, not holdout labels.
        sensitivity = []
        for key in ("geometry", "tag"):
            altered = dict(learned)
            altered["team_weights"] = list(learned["team_weights"])
            altered["team_weights"][("geometry", "tag").index(key)] = 0.
            labels, _, _, _ = predict(arrays, config, samples=0, score_override=calibrated_scores(arrays, config, altered))
            sensitivity.append(labels)
        disagreement = np.mean(np.asarray(sensitivity) != final, axis=0)
        np.savez_compressed(out / "calibrated_diagnostics.npz", posterior=posterior,
                            sampled_histories=histories, source_disagreement=disagreement)
        # Calibration is an explicit development comparison, not automatic model
        # selection. Keep the predeclared label-free submission as the main output.
        final, posterior, histories, diagnostics = base, base_posterior, base_histories, base_diagnostics
        disagreement = base_disagreement
    hashes["blindspot.csv"] = save_submission(out / "blindspot.csv", final)
    queries = choose_queries(posterior, disagreement, arrays, config, args.query_policy)
    save_json(out / "queries.json", queries)
    save_json(out / "query_controls.json", {policy: choose_queries(posterior, disagreement, arrays, config, policy)
                                            for policy in ("random", "uniform", "entropy")})
    receipts = boundary_receipts(final, posterior, arrays, disagreement)
    save_json(out / "boundary_receipts.json", receipts)
    np.savez_compressed(out / "inference.npz", posterior=posterior, sampled_histories=histories,
                        source_disagreement=disagreement)
    gap_records = []
    from games import GAME, half_video_blocks
    for half, start, last in half_video_blocks(GAME):
        missing = ~np.isfinite(arrays["ball"][start:last + 1]).all(1)
        for a, b in intervals(missing):
            low, high = start + a, start + b
            if high - low < 5:
                continue
            sequences = set()
            for history in histories[:, low:high]:
                compact = history[np.r_[True, history[1:] != history[:-1]]]
                sequences.add("".join("HADX"[int(x)] for x in compact))
            gap_records.append({"start_frame": low, "stop_frame": high, "half": half,
                                "seconds": (high - low) / 25., "alternative_state_sequences": sorted(sequences),
                                "sampled_histories_agree": len(sequences) == 1,
                                "min_model_support": float(posterior[low:high].max(1).min()),
                                "maximum_source_disagreement": float(disagreement[low:high].max()),
                                "scope": "Retained state histories only; no claim of physical completeness"})
    save_json(out / "gap_histories.json", gap_records)
    if args.visual_budget:
        from possession.visual import measure
        video = Path(args.data_dir) / "tactical/BAR-ATH_tactical.mp4"
        measurements = measure(video, queries, args.visual_budget)
        save_json(out / "visual_measurements.json", measurements)
        # Measurements stay diagnostic until their link to state changes is validated.
        print("Visual measurements complete; no unsupported contact labels injected", flush=True)
    save_json(out / "scores.json", evaluation)
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unavailable"
    source_hash = hashlib.sha256()
    source_paths = sorted((ROOT / "possession").glob("*.py")) + [Path(__file__).resolve()]
    for path in source_paths:
        source_hash.update(path.name.encode())
        source_hash.update(path.read_bytes())
    manifest = {"version": __version__, "git_revision": revision, "source_sha256": source_hash.hexdigest(),
                "config": config, "input_sha256": feature_meta["input_sha256"], "output_sha256": hashes,
                "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "public_label_training": bool(args.train_public), "query_policy": args.query_policy,
                "visual_measurement_budget": args.visual_budget,
                "scope": "Single-match research prototype. Model marginals not calibrated; sampled agreement not a certificate.",
                "submission_choice": "evidence_v1 (calibration remains a separate development comparison)",
                "gap_histories_count": len(gap_records)}
    save_json(out / "manifest.json", manifest)
    write_report(out / "report.html", final, posterior, histories, disagreement, arrays, queries, evaluation, manifest)
    for name, result in evaluation.items():
        print("{}: macroF1 {:.4f} spellF1 {:.4f} ({})".format(name, result["macro_f1"], result["spell_f1_mean"], result["scope"]), flush=True)
    print("Validated CSV and interactive report:", out.resolve(), flush=True)
    return manifest


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default=str(ROOT / "data"))
    p.add_argument("--out", default=str(ROOT / "data/experiments/blindspot-v1"))
    p.add_argument("--config", default=str(ROOT / "configs/evidence_v1.json"))
    p.add_argument("--ablations", action="store_true")
    p.add_argument("--train-public", action="store_true", help="Fit on first 120s only; evaluate later 170s with 10s buffer")
    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument("--query-policy", choices=("boundary", "random", "uniform", "entropy"), default="boundary")
    p.add_argument("--query-budget", type=int)
    p.add_argument("--visual-budget", type=int, default=0)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
