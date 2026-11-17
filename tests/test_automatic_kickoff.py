import numpy as np
import pytest

from scripts.research.video_ball.export_candidate import automatic_prekickoff_dead


def tag(start, half='2nd Half', kind='Kick Off (Start)'):
    return dict(code='SET PIECES',start_frame=start,tags={'Half':half,'Type':kind})


def test_tag_start_drives_generic_dead_interval_with_no_manual_anchor():
    labels=np.array(['X','H','A','H','H','A','X'])
    new,intervals=automatic_prekickoff_dead(labels,[tag(4)],[(2,1,5)])
    np.testing.assert_array_equal(new,['X','D','D','D','H','A','X'])
    np.testing.assert_array_equal(labels,['X','H','A','H','H','A','X'])
    assert intervals==[[1,4]]


def test_after_goal_and_unknown_halves_do_not_override_start():
    labels=np.array(['H']*8)
    new,intervals=automatic_prekickoff_dead(labels,[tag(5,kind='Kick Off (After Goal)')],[(2,0,7)])
    np.testing.assert_array_equal(new,labels)
    assert not intervals
    with pytest.raises(ValueError):
        automatic_prekickoff_dead(labels,[tag(3),tag(5)],[(2,0,7)])
