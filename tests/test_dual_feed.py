import numpy as np
import pytest

from scripts.research.broadcast.dual_feed import (
    PROTOCOL, align_signal, audit_video, estimate_offset, fuse_ball, paired_cost, remap_teams,
)


def synthetic_pair():
    rng = np.random.default_rng(142)
    teams = np.repeat([0, 1], 4)
    players = rng.uniform([0, 0], [100, 65], size=(180, 8, 2)).astype(np.float32)
    ball = rng.uniform(0, 50, size=(180, 2)).astype(np.float32)
    tactical = dict(players=players, ball=ball, teams=teams,
                    names=['home', 'away'], valid=np.ones(180, bool))
    # Reverse the team ordering and permute slot identities in the second camera.
    order = np.array([7, 5, 4, 6, 1, 3, 0, 2])
    broadcast = dict(players=players[12:162, order].copy(), ball=ball[12:162].copy(),
                     teams=1 - teams[order], names=['away', 'home'],
                     valid=np.ones(150, bool))
    broadcast['players'][35:45] = players[100:110, order]  # A replay on the wrong timeline.
    broadcast['valid'][65:70] = False
    settings = dict(PROTOCOL, coarse_tactical_stride=4, coarse_broadcast_stride=10,
                    refinement_stride=5)
    return tactical, broadcast, settings


def test_global_alignment_recovers_offset_without_frame_seeds_and_rejects_replays():
    tactical, broadcast, settings = synthetic_pair()
    signal, receipts = align_signal(tactical, broadcast, [(1, 10, 175)], settings)
    assert receipts[0]['offset'] == 12
    np.testing.assert_array_equal(signal['ball'][32:42], tactical['ball'][32:42])
    assert not signal['accepted_view'][47:57].any()
    assert not signal['accepted_view'][77:82].any()
    assert not signal['accepted_view'][:10].any()
    assert not signal['accepted_view'][175:].any()


def test_absent_players_are_not_zero_cost_and_slots_have_no_identity():
    p = np.array([[[1., 2.], [3., 4.], [5., 6.], [7., 8.]]])
    teams = np.array([0, 0, 1, 1])
    cost, count = paired_cost(p[:, [1, 0, 3, 2]], teams, p, teams)
    assert cost[0] == 0 and count[0] == 4
    absent, _ = paired_cost(p, teams, np.full_like(p, np.nan), teams)
    assert absent[0] == 5
    empty, count = paired_cost(np.full_like(p, np.nan), teams, p, teams)
    assert np.isnan(empty[0]) and count[0] == 0


def test_team_name_matching_handles_different_metadata_order_and_refuses_wrong_match():
    np.testing.assert_array_equal(remap_teams(['away', 'home'], ['home', 'away'],
                                            np.array([0, 1, 0])), [1, 0, 1])
    with pytest.raises(ValueError, match='same two teams'):
        remap_teams(['home', 'other'], ['home', 'away'], np.array([0, 1]))


def test_unrelated_recordings_fail_closed():
    tactical, broadcast, settings = synthetic_pair()
    broadcast['players'][:] = 10000
    with pytest.raises(ValueError, match='Insufficient geometric matches'):
        estimate_offset(broadcast['players'], broadcast['teams'], broadcast['valid'],
                        tactical['players'], tactical['teams'], 10, 175, settings)


def test_fusion_preserves_observations_and_blocks_invalid_views_and_impossible_motion():
    original = np.array([[0, 0], [np.nan, np.nan], [np.nan, np.nan], [.3, 0]], np.float32)
    signal = dict(ball=np.array([[99, 99], [.1, 0], [.2, 0], [99, 99]], np.float32),
                  accepted_view=np.array([True, True, True, True]))
    fused, source, _ = fuse_ball(original, signal, [(0, 4)], 10.)
    np.testing.assert_allclose(fused, [[0, 0], [.1, 0], [.2, 0], [.3, 0]])
    np.testing.assert_array_equal(source, [1, 2, 2, 1])
    signal['accepted_view'][1] = False
    signal['ball'][2] = [99, 99]
    fused, _, _ = fuse_ball(original, signal, [(0, 4)], 10.)
    assert not np.isfinite(fused[1:3]).any()


def test_tactical_fallback_rejects_conflicting_cross_source_junction():
    original = np.array([[0, 0], [np.nan, np.nan], [np.nan, np.nan], [.3, 0]], np.float32)
    signal = dict(ball=np.array([[0, 0], [.1, 0], [np.nan, np.nan], [.3, 0]], np.float32),
                  accepted_view=np.ones(4, bool))
    prior = np.array([[0, 0], [.12, 0], [50, 0], [.3, 0]], np.float32)
    fused, source, _ = fuse_ball(original, signal, [(0, 4)], 10., prior)
    assert fused[1, 0] == signal['ball'][1, 0]
    assert np.isnan(fused[2]).all() and source[2] == 0
    prior[0] = [1, 0]
    with pytest.raises(ValueError, match='changed original'):
        fuse_ball(original, signal, [(0, 4)], 10., prior)


def test_hypotheses_cannot_bridge_halftime_without_an_anchor():
    original = np.full((6, 2), np.nan, np.float32)
    signal = dict(ball=np.zeros((6, 2), np.float32), accepted_view=np.ones(6, bool))
    fused, source, _ = fuse_ball(original, signal, [(0, 3), (3, 6)], 10.)
    assert not np.isfinite(fused).any() and not source.any()


def test_video_mismatch_is_explicit_and_never_silently_approved_for_pixels(monkeypatch):
    import cv2

    class Capture:
        released = False

        def isOpened(self):
            return True

        def get(self, key):
            return {cv2.CAP_PROP_FRAME_COUNT: 4, cv2.CAP_PROP_FPS: 25,
                    cv2.CAP_PROP_FRAME_WIDTH: 8, cv2.CAP_PROP_FRAME_HEIGHT: 6}[key]

        def set(self, key, value):
            pass

        def read(self):
            return True, np.zeros((6, 8, 3), np.uint8)

        def release(self):
            self.released = True

    capture = Capture()
    monkeypatch.setattr(cv2, 'VideoCapture', lambda path: capture)
    with pytest.raises(ValueError, match='frame count or frame rate'):
        audit_video('synthetic.mp4', 5)
    assert capture.released
    result = audit_video('synthetic.mp4', 5, require_exact=False)
    assert result['frame_count_difference'] == -1
    assert not result['pixel_inference_timeline_verified']
    assert not result['used_for_new_pixel_inference']


def test_external_sensor_simulation_is_deterministic_and_does_not_modify_inputs():
    from scripts.research.broadcast.benchmark_dual_feed import synthetic_views

    tactical, _, _ = synthetic_pair()
    players, ball = tactical['players'].copy(), tactical['ball'].copy()
    a, sa = synthetic_views(players, tactical['teams'], ball, 713)
    b, sb = synthetic_views(players, tactical['teams'], ball, 713)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(sa['ball'], sb['ball'])
    np.testing.assert_array_equal(players, tactical['players'])
    np.testing.assert_array_equal(ball, tactical['ball'])


def test_external_scorer_keeps_spells_separate_across_halves():
    from scripts.research.broadcast.benchmark_dual_feed import evaluate

    parts = [np.array(['H', 'H']), np.array(['H', 'H'])]
    scores = evaluate(parts, parts)
    assert scores['n_gt_spells'] == 2
    assert scores['n_pred_spells'] == 2
    assert scores['spell_f1_mean'] == 1.
