import numpy as np
from possession.frozen_boundary_transfer import transition_targets,balance_weights,calibrated_probability


def test_boundary_targets_use_one_half_clock_and_exclude_unknown_neighbors():
    labels=np.array(list('HHAAAXDDHH'))
    np.testing.assert_array_equal(transition_targets(labels,1),[-100,0,1,0,0,-100,-100,0,1,0])
    np.testing.assert_array_equal(transition_targets(labels,2),[-100,1,0,1,1])
    assert transition_targets(np.array(['H']))[0]==-100
    # Separate halves each exclude their first frame, even if the team differs.
    assert transition_targets(np.array(['A','A']),1)[0]==-100


def test_analytic_correction_recovers_unbalanced_probability():
    target=np.array([0,0,0,1]);sampling=np.array([3.,3.,3.,1.])
    weights,shift=balance_weights(target,sampling)
    assert np.isclose(weights[target==0].sum(),weights[target==1].sum())
    truth=np.array([.01,.2,.5,.9]);balanced_logits=np.log(truth/(1-truth))+shift
    np.testing.assert_allclose(calibrated_probability(balanced_logits,shift),truth)
