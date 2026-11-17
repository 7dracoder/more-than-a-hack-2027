import copy
from scripts.research.spatial_memory.train_real_broadcast_memory import joint_selection


def example():
    strong = {k:{'spell_f1_mean':v} for k, v in [('native', .86), ('masked', .53), ('mixed', .62)]}
    weak = {'all_known':{'conditional_ha_macro_f1':.8, 'weak_ha_accuracy':.8},
        'ball_missing':{'conditional_ha_accuracy':.75}}
    initial = {'strong_scores':copy.deepcopy(strong), 'strong_mean':.67}
    reference = {'all_known':{'weak_ha_accuracy':.75}, 'ball_missing':{'conditional_ha_accuracy':.66}}
    return strong, weak, initial, reference


def test_joint_gain_is_eligible_when_both_domains_remain_above_guards():
    result = joint_selection(*example())
    assert result['eligible'] and all(result['guards'].values())
    assert abs(result['objective'] - .735) < 1e-12


def test_high_joint_score_cannot_override_either_weak_domain_guard():
    strong, weak, initial, reference = example()
    weak['all_known']['weak_ha_accuracy'] = .7
    result = joint_selection(strong, weak, initial, reference)
    assert not result['eligible'] and not result['guards']['weak_ha']
    weak['all_known']['weak_ha_accuracy'] = .8
    weak['ball_missing']['conditional_ha_accuracy'] = .5
    result = joint_selection(strong, weak, initial, reference)
    assert not result['eligible'] and not result['guards']['weak_missing_ha']


def test_weak_gain_cannot_override_native_or_strong_spell_degradation():
    strong, weak, initial, reference = example()
    strong['native']['spell_f1_mean'] = .8
    result = joint_selection(strong, weak, initial, reference)
    assert not result['eligible'] and not result['guards']['native']
    strong['native']['spell_f1_mean'] = .86
    strong['masked']['spell_f1_mean'] = .2
    result = joint_selection(strong, weak, initial, reference)
    assert not result['eligible'] and not result['guards']['strong_mean']
