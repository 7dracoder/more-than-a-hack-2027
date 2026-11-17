import numpy as np
from possession.whistle_transfer import temporal_features,point_targets,frame_probabilities


def test_temporal_context_replicates_edges_without_recording_wrap():
    x=np.arange(3)[:,None]
    np.testing.assert_array_equal(temporal_features(x),[[0,0,1],[0,1,2],[1,2,2]])


def test_point_labels_ignore_uncertain_neighborhoods():
    result=point_targets(np.array([0.,1.8,2.,2.5,4.,6.]),[2.])
    np.testing.assert_array_equal(result,[0,1,1,-100,0,0])


def test_native_bin_clock_preserves_boundary_phase():
    p=frame_probabilities(np.arange(40),1050)
    assert p[0]==0 and p[23]==0 and p[24]==1
    assert p[1023]==32 and p[1024]==32 and p[1032]==32
