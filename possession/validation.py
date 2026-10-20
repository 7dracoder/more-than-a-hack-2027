"""Possession-grouped development split and label-independent query evaluation."""

import numpy as np

from possession.data import intervals


def grouped_split(labels, target=14185, embargo=250, context=50):
    """Drop the entire run containing target; never divide a labelled state run.

    The embargo is rounded OUT to full run boundaries. Consecutive equal labels
    form a group; no random frame split or training on X is permitted.
    """
    labels = np.asarray(labels)
    if not 0 < target < len(labels) or embargo < context * 2:
        raise ValueError("Invalid target or embargo shorter than twice feature context")
    edges = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
    runs = [(int(a), int(b), str(labels[a])) for a, b in zip(edges[:-1], edges[1:]) if labels[a] != 'X']
    train_runs = [(a, b, state) for a, b, state in runs if b <= target]
    if not train_runs:
        raise ValueError("No complete training runs before target")
    train_stop = train_runs[-1][1]
    test_runs = [(a, b, state) for a, b, state in runs if a >= train_stop + embargo]
    if not test_runs:
        raise ValueError("No complete test runs after embargo")
    train = np.zeros(len(labels), dtype=bool)
    test = train.copy()
    for a, b, _ in train_runs:
        train[a:b] = True
    for a, b, _ in test_runs:
        test[a:b] = True
    return train, test, {
        "kind": "whole_state_run_temporal_development_split", "target_frame": target,
        "minimum_embargo_frames": embargo, "feature_context_radius_frames": context,
        "train_runs": train_runs, "test_runs": test_runs,
        "train_frames": int(train.sum()), "test_frames": int(test.sum()),
        "actual_gap_frames": test_runs[0][0] - train_stop,
        "warning": "Previously examined public same-match data; NOT a fresh independent test. No hyperparameter selection here."}


def query_benchmark(posterior, disagreement, arrays, config, truth, prediction, blocks):
    """Error enrichment AFTER selection. No correction or improvement is simulated.

    Queries are clipped and nonoverlapping within each policy. Errors count only
    labelled frames; bootstrap controls vary random seeds, not model parameters.
    """
    from possession.model import choose_queries

    truth, prediction = np.asarray(truth), np.asarray(prediction)
    valid = truth != 'X'
    wrong = (truth != prediction) & valid
    transitions = np.flatnonzero((truth[1:] != truth[:-1]) & valid[1:] & valid[:-1]) + 1
    global_error = float(wrong.sum() / max(1, valid.sum()))

    def measure(policy, seed):
        settings = dict(config, seed=seed)
        queries = choose_queries(posterior, disagreement, arrays, settings, policy, blocks=blocks)
        selected = np.zeros(len(truth), dtype=bool)
        for query in queries:
            selected[query['start_frame']:query['stop_frame']] = True
        graded = selected & valid
        errors = int((wrong & selected).sum())
        rate = errors / max(1, int(graded.sum()))
        return {"queries": queries, "queries_count": len(queries), "review_seconds": float(selected.sum() / 25),
                "graded_frames": int(graded.sum()), "error_frames": errors, "error_rate": rate,
                "error_enrichment": rate / global_error if global_error else None,
                "fraction_of_all_errors_covered": errors / max(1, int(wrong.sum())),
                "true_boundaries_in_clips": int(selected[transitions].sum()),
                "true_boundaries_total": len(transitions)}

    result = {policy: measure(policy, config['seed']) for policy in ('boundary', 'entropy', 'uniform', 'random')}
    controls = [measure('random', config['seed'] + i) for i in range(30)]
    values = [item['error_rate'] for item in controls]
    return {"scope": "Public development window only; ranking never receives truth labels. Error discovery is not correction or time saved.",
            "overall_error_rate": global_error, "policies": result,
            "random_30_seeds": {"mean_error_rate": float(np.mean(values)),
                                "p05_p95_error_rate": np.quantile(values, [.05, .95]).tolist(),
                                "seed_start": config['seed'], "n": 30}}


def class_diagnostics(truth, prediction):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    valid = truth != 'X'
    table = {}
    for state in ('H', 'A', 'D'):
        actual, predicted = (truth == state) & valid, (prediction == state) & valid
        tp = int((actual & predicted).sum())
        fp, fn = int((predicted & ~actual).sum()), int((actual & ~predicted).sum())
        table[state] = {"true_frames": int(actual.sum()), "predicted_frames": int(predicted.sum()),
                        "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn),
                        "f1": 2 * tp / max(1, 2 * tp + fp + fn)}
    missed = (truth == 'D') & (prediction != 'D')
    false = valid & (truth != 'D') & (prediction == 'D')
    def longest(mask):
        return [{"start_frame": a, "stop_frame": b, "seconds": (b-a)/25}
                for a, b in sorted(intervals(mask), key=lambda x: x[1]-x[0], reverse=True)[:12]]
    return {"classes": table, "longest_missed_stoppages": longest(missed),
            "longest_false_stoppages": longest(false)}
