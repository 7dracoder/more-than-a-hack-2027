import numpy as np
from scripts.research.spatial_memory.audit_elastic_timing import neighborhood


def test_neighborhood_reports_smallest_agreeing_radius():
    truth=np.full(70,'A');truth[35]='H'
    assert neighborhood(truth,35,'H')=='exact'
    assert neighborhood(truth,34,'H')=='agrees_within_1_frames'
    assert neighborhood(truth,30,'H')=='agrees_within_5_frames'
    assert neighborhood(truth,10,'H')=='agrees_within_25_frames'
    assert neighborhood(truth,0,'H')=='no_agreement_within_25_frames'


def test_half_edge_does_not_wrap_into_other_end():
    truth=np.full(80,'A');truth[-1]='H'
    assert neighborhood(truth,0,'H')=='no_agreement_within_25_frames'
    truth[0]='H';truth[-1]='A'
    assert neighborhood(truth,79,'H')=='no_agreement_within_25_frames'
