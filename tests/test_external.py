"""External supervision isolation, provider alignment and temporal inference parity."""
import tempfile
from pathlib import Path
import unittest

import numpy as np

from possession.external import build_features, external_labels, read_sportec, swap_features
from possession.decode import infer, transition_costs


class ExternalTests(unittest.TestCase):
    def test_opened_test_matches_cannot_be_reused_from_another_run(self):
        from possession.evaluation_guard import reserve_test_cohort
        with tempfile.TemporaryDirectory() as tmp:
            registry=Path(tmp)/'registry.json'
            reserve_test_cohort(registry,{'match-a':'hash-a','match-b':'hash-b'},'model-a','run-a')
            with self.assertRaisesRegex(ValueError,'already opened'):
                reserve_test_cohort(registry,{'match-b':'hash-b'},'model-b','run-b')
            reserve_test_cohort(registry,{'match-c':'hash-c'},'model-c','run-c')

    def test_missing_ball_augmentation_is_reproducible_without_label_inputs(self):
        from scripts.prepare_external_gaps import missing_intervals
        a=missing_intervals(10000,12)
        b=missing_intervals(10000,12)
        np.testing.assert_array_equal(a,b)
        self.assertGreater(a.mean(),.2); self.assertLess(a.mean(),.7)

    def test_historical_training_is_not_enabled_by_default(self):
        from possession.policy import require_historical_opt_in
        with self.assertRaisesRegex(ValueError, 'October 23'):
            require_historical_opt_in(False)
        require_historical_opt_in(True)

    def test_same_match_annotation_plan_is_disabled(self):
        from scripts.prepare_blind_validation import main
        with self.assertRaisesRegex(SystemExit, 'no hand labels'):
            main()

    def test_unknown_live_possession_stays_ungraded_and_dead_overrides_owner(self):
        result = external_labels(['1','1','0','1',''], ['1','2','1','0','1'])
        self.assertEqual(result.tolist(), ['H','A','D','X','X'])

    def test_ball_state_flags_cannot_change_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            meta = root/'meta.xml'; track = root/'track.xml'
            meta.write_text('<Root><General MatchId="DFL-MAT-TEST" HomeTeamId="H" GuestTeamId="A"/><Environment PitchX="105" PitchY="68"/></Root>')
            objects = ''.join('<FrameSet GameSection="firstHalf" TeamId="'+side+'" PersonId="'+side+str(i)+'">'+
                              ''.join(f'<Frame N="{f}" X="{i}" Y="{f/100}"/>' for f in range(10, 40))+'</FrameSet>'
                              for side in ('H','A') for i in range(3))
            def source(status, owner):
                return '<Root>'+objects+'<FrameSet GameSection="firstHalf" TeamId="BALL" PersonId="ball">'+''.join(
                    f'<Frame N="{f}" X="0" Y="{f/100}" BallStatus="{status}" BallPossession="{owner}"/>'
                    for f in range(10,40))+'</FrameSet></Root>'
            track.write_text(source('1','1')); a = read_sportec(meta, track)[0]
            track.write_text(source('0','2')); b = read_sportec(meta, track)[0]
            self.assertTrue((a.labels == 'H').all()); self.assertTrue((b.labels == 'D').all())
            xa, names = build_features(a.players, a.teams, a.ball, a.pitch)
            xb, other = build_features(b.players, b.teams, b.ball, b.pitch)
            np.testing.assert_array_equal(xa, xb)
            self.assertEqual(names, other)
            self.assertEqual(a.frames.tolist(), list(range(10,40)))
            np.testing.assert_allclose(a.ball[:,0], 52.5)

    def test_team_swap_and_joint_slot_permutation(self):
        rng = np.random.default_rng(1)
        p = rng.normal(size=(90,8,2)) + [50,34]
        b = rng.normal(size=(90,2)) + [50,34]
        teams = np.repeat([0,1],4)
        x,names = build_features(p, teams, b, (105,68))
        other,_ = build_features(p, 1-teams, b, (105,68))
        np.testing.assert_allclose(swap_features(x,names), other, atol=1e-6)
        order = rng.permutation(8)
        perm,_ = build_features(p[:,order], teams[order], b, (105,68))
        np.testing.assert_allclose(x, perm, atol=1e-6)
        np.testing.assert_array_equal(swap_features(swap_features(x,names),names),x)

    def test_missing_ball_does_not_create_contact_or_out_of_play(self):
        p = np.ones((60,8,2)); teams = np.repeat([0,1],4)
        x,names = build_features(p,teams,np.full((60,2),np.nan),(105,68))
        self.assertTrue(np.isfinite(x).all())
        for name in ('ball_seen','home_contact','away_contact','home_last_contact','away_last_contact','ball_outside_known'):
            self.assertTrue((x[:,names.index(name)] == 0).all())


class TemporalTests(unittest.TestCase):
    def test_saved_ensemble_matches_averaged_member_probabilities(self):
        import torch
        from possession.temporal import PossessionTCN,PossessionBiGRU,ProbabilityEnsemble,symmetric_probability
        torch.set_num_threads(2); torch.manual_seed(4)
        names=['home_a','away_a','ball_seen']
        x=np.random.default_rng(4).normal(size=(100,3)).astype(np.float32)
        mean,scale=np.zeros(3,np.float32),np.ones(3,np.float32)
        first=PossessionTCN(3,8,(1,2)); second=PossessionBiGRU(3,8)
        pa=symmetric_probability(first,x,names,mean,scale,'cpu',chunk=20)
        pb=symmetric_probability(second,x,names,mean,scale,'cpu',chunk=20)
        combined=symmetric_probability(ProbabilityEnsemble([first,second],[.25,.75]),x,names,mean,scale,'cpu',chunk=20)
        np.testing.assert_allclose(combined,.25*pa+.75*pb,atol=1e-6)

    def test_recurrent_inference_does_not_reset_at_chunk_boundaries(self):
        import torch
        from possession.temporal import PossessionBiGRU,symmetric_probability
        torch.set_num_threads(2); torch.manual_seed(2)
        x=np.random.default_rng(2).normal(size=(100,3)).astype(np.float32)
        names=['home_a','away_a','ball_seen']
        model=PossessionBiGRU(3,8)
        args=(model,x,names,np.zeros(3,np.float32),np.ones(3,np.float32),'cpu')
        a=symmetric_probability(*args,chunk=16)
        b=symmetric_probability(*args,chunk=1000)
        np.testing.assert_allclose(a,b,atol=1e-7)

    def test_equal_run_weighting_keeps_unknown_frames_out(self):
        from possession.temporal import training_weights
        labels=np.array(list('H'*10+'A'*40+'X'*5))
        weight=training_weights(labels,power=1.)
        self.assertAlmostEqual(weight[:10].sum(),weight[10:50].sum(),places=5)
        self.assertTrue((weight[50:]==0).all())

    def test_training_pool_keeps_half_boundaries_out_of_supervision(self):
        from scripts.run_local import training_table
        halves = [dict(x=np.ones((3,2)),y=np.array(list('HAD'))),
                  dict(x=np.full((3,2),2.),y=np.array(list('DAH')))]
        x,y = training_table(halves)
        self.assertEqual(len(x),len(y))
        self.assertEqual(y.tolist(),list('HADXDAHX'))
        np.testing.assert_array_equal(x[y!='X'],np.vstack([h['x'] for h in halves]))

    def test_fast_decoder_equals_exact_reference(self):
        from possession.temporal import viterbi_labels
        rng = np.random.default_rng(99)
        for _ in range(8):
            p = rng.dirichlet([1,1,1],size=100)
            a = viterbi_labels(p,3,4)
            path,_,_ = infer(np.log(p),transition_costs(3,4),samples=0)
            np.testing.assert_array_equal(a,np.array(['H','A','D'])[path])

    def test_overlapping_inference_has_no_chunk_seams_and_is_symmetric(self):
        import torch
        from possession.temporal import PossessionTCN, symmetric_probability
        torch.set_num_threads(2); torch.manual_seed(1)
        rng = np.random.default_rng(1)
        names = ['home_a','away_a','ball_seen']
        x = rng.normal(size=(180,3)).astype(np.float32)
        model = PossessionTCN(3,8,(1,2,4))
        mean,scale = np.zeros(3,np.float32),np.ones(3,np.float32)
        p = symmetric_probability(model,x,names,mean,scale,'cpu',chunk=32)
        whole = symmetric_probability(model,x,names,mean,scale,'cpu',chunk=1000)
        swapped = symmetric_probability(model,swap_features(x,names),names,mean,scale,'cpu',chunk=32)
        np.testing.assert_allclose(p,whole,atol=1e-6)
        np.testing.assert_allclose(p,swapped[:,[1,0,2]],atol=1e-6)

    def test_competition_match_and_overlapping_splits_are_rejected(self):
        from scripts.run_local import validate_splits
        with self.assertRaises(ValueError):
            validate_splits(dict(train_matches=['BAR-ATH'],validation_matches=['a'],test_matches=['b']))
        with self.assertRaises(ValueError):
            validate_splits(dict(train_matches=['a'],validation_matches=['a'],test_matches=['b']))


if __name__ == '__main__':
    unittest.main()
