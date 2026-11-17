import numpy as np
import pytest
from evaluation import possession_metrics as pm
from possession.external_spell_uncertainty import spell_information,paired_draws


def test_each_paired_margin_matches_official_bootstrap_exactly():
    t = np.array(list('HHHHDAAAADHHHHDAAAADHHHHDAAAAD'))
    first = np.array(list('HHHHDAAAADHHHHDAAAADHHHHDAAAAD'))
    second = np.array(list('HHXXDAAAADDDHHDAAHHDHHHHDAADDD'))
    info = [spell_information([t],[p]) for p in [first,second]]
    assert info[0][2] == info[1][2]
    flags = np.asarray([[item[0] for item in info]])
    counts = np.asarray([[item[1] for item in info]])
    draws,_ = paired_draws(flags,counts,n_draws=200,seed=0)
    for i,pred in enumerate([first,second]):
        tt,pp = pm.label_codes(np.concatenate([t,['X']])),pm.label_codes(np.concatenate([pred,['X']]))
        runs = pm.valid_runs(tt)
        expected = pm.bootstrap_spell_f1(tt,pm.fill_abstentions(pp,runs),runs,200)
        np.testing.assert_array_equal(np.percentile(draws[0,i],[5,95]),expected)


def test_identical_models_have_zero_paired_difference():
    rng = np.random.default_rng(12)
    one = rng.random((3,5,40)) > .4
    flags = np.stack([one,one],axis=1)
    counts = np.full((3,2),50)
    _,delta = paired_draws(flags,counts,100)
    np.testing.assert_array_equal(delta,np.zeros(100))


def test_invalid_pairing_shape_rejected():
    with pytest.raises(ValueError):
        paired_draws(np.ones((3,3,5,40),bool),np.ones((3,3)))
