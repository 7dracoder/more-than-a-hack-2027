import numpy as np
from possession.external_object_metrics import matches,summary


def test_duplicate_predictions_and_negative_image_affect_precision_and_ap():
    positive=dict(truth=np.array([[0,0,10,10]]),boxes=np.array([[0,0,10,10],[0,0,10,10]]),confidence=np.array([.8,.7]))
    negative=dict(truth=np.empty((0,4)),boxes=np.array([[0,0,10,10]]),confidence=np.array([.9]))
    result=summary([positive,negative])
    assert (result['tp'],result['fp'],result['fn'])==(1,2,0)
    assert result['ap50']==.5 and result['small_recall']==1.


def test_matching_never_credits_one_prediction_to_two_objects():
    flags,assigned=matches([[0,0,10,10]],[.9],[[0,0,10,10],[1,1,11,11]])
    assert flags.tolist()==[True] and (assigned>=0).sum()==1
    result=summary([dict(truth=np.array([[0,0,10,10]]),boxes=np.empty((0,4)),confidence=np.empty(0))])
    assert result['ap50']==0 and result['fn']==1 and result['recall']==0
