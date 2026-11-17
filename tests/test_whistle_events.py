import numpy as np
from possession.whistle_events import event_score,extract_events


def test_point_matching_maximizes_cardinality_without_double_counting():
    result=event_score([0,.5],[.4,.6],.5)
    assert result['tp']==2 and result['f1']==1
    assert event_score([1],[.9,1.1],.5)['fp']==1
    assert event_score([1,1.1],[1.05],.5)['fn']==1


def test_gap_merge_keeps_one_strongest_event_and_refractory_is_deterministic():
    x=np.zeros(200);x[30:35]=.8;x[40:45]=.9;x[65:70]=.85
    np.testing.assert_allclose(extract_events(x,.5),[.4])
    np.testing.assert_allclose(extract_events(x,.5,refractory=.1),[.4,.65])


def test_empty_scores_and_unmatched_events():
    assert len(extract_events(np.zeros(100),.5))==0
    assert event_score([1,2],[],.5)['fn']==2
    assert event_score([],[1],.5)['fp']==1
