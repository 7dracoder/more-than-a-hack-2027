import numpy as np
import pytest
from scripts.research.spatial_memory.audit_broadcast_transfer import weak_counts, summarize_counts


def examples():
    p = np.array([[.9, .05, .05, .1], [.2, .3, .5, .1], [.1, .1, .8, .1], [.1, .8, .1, .1]])
    return p, np.array(list('HDDA')), np.array(list('HAXH'))


def test_conditional_ha_accuracy_does_not_hide_dead_predictions_in_main_accuracy():
    p, predicted, truth = examples()
    counts = weak_counts(p, predicted, truth, np.ones(4, bool))
    result = summarize_counts([counts])
    assert result['frames'] == 3
    assert result['weak_ha_accuracy'] == 1 / 3
    assert result['conditional_ha_accuracy'] == 2 / 3
    assert result['dead_prediction_rate_on_weak_ha'] == 1 / 3
    np.testing.assert_allclose(result['conditional_ha_macro_f1'], 2 / 3)


def test_unknown_provider_targets_never_become_dead_truth():
    p, predicted, _ = examples()
    counts = weak_counts(p, predicted, np.full(4, 'X'), np.ones(4, bool))
    result = summarize_counts([counts])
    assert result['frames'] == 0
    assert result['weak_ha_accuracy'] is None and result['dead_prediction_rate_on_weak_ha'] is None


def test_count_aggregation_equals_pooled_samples():
    p, predicted, truth = examples()
    split = [weak_counts(p[a:b], predicted[a:b], truth[a:b], np.ones(b - a, bool)) for a, b in [(0, 1), (1, 4)]]
    pooled = weak_counts(p, predicted, truth, np.ones(4, bool))
    a, b = summarize_counts(split), summarize_counts([pooled])
    assert a.pop('conditional_ha_nll') == pytest.approx(b.pop('conditional_ha_nll'), abs=1e-14)
    assert a == b
