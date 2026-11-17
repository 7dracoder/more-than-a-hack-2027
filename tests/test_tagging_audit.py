import numpy as np

from scripts.research.video_ball.audit_tagging import phase_evidence


def event(code, team, start=2, stop=8):
    return dict(code=code, tags={'Team': team}, start_frame=start, end_frame=stop)


def test_opponent_defense_is_same_cue_as_offense_not_independent_vote():
    events = [event('BUILD UP', 'Barcelona'), event('DEFENDING IN ATTACKING THIRD', 'Atletic Club')]
    owner, count, intervals, unknown = phase_evidence(events, 10)
    assert len(intervals) == 1 and not unknown
    assert len(next(iter(intervals.values()))) == 2
    np.testing.assert_array_equal(owner, [-1, -1, 0, 0, 0, 0, 0, 0, -1, -1])
    assert (count[2:8, 0] == 1).all() and not count[:, 1].any()


def test_conflicting_owners_remain_unknown():
    events = [event('PROGRESSION', 'Barcelona'), event('LONG BALL', 'Athletic Club', 4, 6)]
    owner, count, _, _ = phase_evidence(events, 10)
    assert (owner[4:6] == -1).all() and (count[4:6] == 1).all()
    assert owner[3] == 0 and owner[6] == 0


def test_spatial_and_restart_tags_do_not_claim_live_possession():
    events = [event(code, 'Barcelona') for code in ('SET PIECES', 'GOALS', 'PLAYERS IN THE BOX', 'BALL IN THE BOX', 'BALL IN FINAL THIRD')]
    owner, count, intervals, _ = phase_evidence(events, 10)
    assert (owner == -1).all() and not count.any() and not intervals


def test_intervals_are_half_open_and_clipped_and_unknown_team_is_reported():
    events = [event('LONG BALL', 'Barcelona', -2, 3), event('LONG BALL', 'N/A'),
              event('PROGRESSION', 'Atletic Club', 9, 20)]
    owner, _, _, unknown = phase_evidence(events, 10)
    assert owner[2] == 0 and owner[3] == -1 and owner[9] == 1
    assert unknown == {'n/a': 1}
