import unittest,json,tempfile
from pathlib import Path
import numpy as np
from possession.skillcorner import weak_target,coordinates,resample_coordinates,roster,load_match


class SkillCornerAdapterTests(unittest.TestCase):
    def test_period_frame_alignment_preserves_unobserved_gaps_as_unknown(self):
        metadata=dict(home_team={'id':1},away_team={'id':2},players=[{'id':31,'team_id':1},{'id':32,'team_id':2}],
            match_periods=[{'period':1,'start_frame':10,'end_frame':12},{'period':2,'start_frame':20,'end_frame':22}])
        rows=[dict(frame=0,period=None)]
        for frame,period,group in ((10,1,'home team'),(12,1,'away team'),(20,2,'away team'),(22,2,'home team')):
            rows.append(dict(frame=frame,period=period,possession={'group':group},player_data=[dict(player_id=31,x=1.,y=2.,is_detected=True)],ball_data=dict(x=3.,y=4.,is_detected=True)))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'tracking.jsonl';path.write_text('\n'.join(json.dumps(row) for row in rows))
            halves,teams=load_match(metadata,path)
            np.testing.assert_array_equal(teams,[0,1]);np.testing.assert_array_equal(halves[1]['labels'],['H','X','A'])
            np.testing.assert_array_equal(halves[2]['labels'],['A','X','H']);self.assertTrue(np.isnan(halves[1]['native'][0][1]).all())
            rows[-1]['frame']=23;path.write_text('\n'.join(json.dumps(row) for row in rows))
            with self.assertRaisesRegex(ValueError,'bounds disagree'):load_match(metadata,path)

    def test_unknown_possession_never_becomes_dead_ball(self):
        self.assertEqual(weak_target(None),'X');self.assertEqual(weak_target({'group':None}),'X')
        self.assertEqual(weak_target({'group':'home team'}),'H');self.assertEqual(weak_target({'group':'away team'}),'A')
        with self.assertRaises(ValueError):weak_target({'group':'unrecognized'})

    def test_targets_and_provider_identity_codes_cannot_enter_coordinates(self):
        row=dict(possession={'group':'home team','player_id':91},player_data=[dict(player_id=91,x=1.,y=2.,is_detected=True)],ball_data=dict(x=3.,y=4.,z=8.,is_detected=True))
        a,b=coordinates(row,{91:0});row['possession']={'group':'away team','player_id':999};row['ball_data']['z']=100
        c,d=coordinates(row,{91:0});np.testing.assert_array_equal(a,c);np.testing.assert_array_equal(b,d)
        row['player_data'][0]['player_id']=9001
        e,f=coordinates(row,{9001:0});np.testing.assert_array_equal(a,e);np.testing.assert_array_equal(b,f)

    def test_detected_view_masks_extrapolated_positions_only(self):
        row=dict(player_data=[dict(player_id=1,x=1.,y=2.,is_detected=False)],ball_data=dict(x=3.,y=4.,is_detected=False))
        a,b=coordinates(row,{1:0});c,d=coordinates(row,{1:0},True)
        self.assertTrue(np.isfinite(a).all() and np.isfinite(b).all());self.assertTrue(np.isnan(c).all() and np.isnan(d).all())
        row['player_data'][0]['is_detected']=True;c,d=coordinates(row,{1:0},True)
        np.testing.assert_array_equal(a,c);self.assertTrue(np.isnan(d).all())

    def test_linear_motion_preserves_metres_seconds_and_source_samples(self):
        source=np.column_stack([np.arange(11,dtype=np.float32)*.5,np.zeros(11,np.float32)])
        result,low=resample_coordinates(source)
        self.assertEqual(len(result),26);np.testing.assert_allclose(result[:,0],np.arange(26)*.2,atol=1e-6)
        np.testing.assert_array_equal(result[::5],source[::2]);np.testing.assert_array_equal(low[::5],np.arange(0,11,2))

    def test_interpolation_never_bridges_missing_points_or_speed_jumps(self):
        source=np.array([[0.,0.],[np.nan,np.nan],[1.,0.],[90.,0.]],np.float32)
        result,_=resample_coordinates(source)
        self.assertTrue(np.isnan(result[1:5]).all());np.testing.assert_array_equal(result[5],[1.,0.])
        self.assertTrue(np.isnan(result[6:]).all())
        constant=np.array([[0.,0.],[.1,0.],[.2,0.]],np.float32)[:,None,:]
        result,_=resample_coordinates(constant);self.assertEqual(result.shape,(6,1,2))

    def test_unknown_or_duplicate_player_and_roster_teams_are_rejected(self):
        row=dict(player_data=[dict(player_id=3,x=1.,y=2.)])
        with self.assertRaises(ValueError):coordinates(row,{1:0})
        row['player_data']*=2
        with self.assertRaises(ValueError):coordinates(row,{3:0})
        with self.assertRaises(ValueError):roster(dict(home_team={'id':1},away_team={'id':2},players=[{'id':3,'team_id':8}]))
