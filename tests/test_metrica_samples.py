import tempfile
from pathlib import Path
import unittest
import numpy as np
from possession.metrica_samples import Event, partial_labels, load_events, controlled, load_tracking, load_csv_side


def event(i, team=0, kind='PASS', start=10, stop=20, sender='p1', receiver='p2', subtype='', period=1):
    return Event(i, period, team, kind, subtype, start, stop, sender, receiver, (.5, .5))


class MetricaLabelTests(unittest.TestCase):
    roster = {'p1': 0, 'p2': 0, 'p3': 1, 'p4': 1}

    def label(self, events, **kw):
        return partial_labels(np.arange(1, 151), 1, events, self.roster, **kw)

    def test_no_events_or_event_absence_never_implies_dead(self):
        labels, _ = self.label([])
        self.assertTrue(np.all(labels == 'X'))
        labels, _ = self.label([event(0)])
        self.assertFalse(np.any(labels == 'D'))
        self.assertEqual(set(np.flatnonzero(labels == 'H')+1), set(range(12, 18)))

    def test_head_clearance_offside_and_opponent_recipient_are_not_control_targets(self):
        for subtype in ('HEAD', 'OFFSIDE', 'CLEARANCE', 'HEAD-CLEARANCE'):
            self.assertFalse(controlled(event(0, subtype=subtype), self.roster))
        self.assertFalse(controlled(event(0, receiver='p3'), self.roster))
        self.assertTrue(controlled(event(0), self.roster))

    def test_same_receiver_bridge_requires_close_gap_and_no_hazard(self):
        a = event(0); b = event(1, start=30, stop=40, sender='p2', receiver='p1')
        labels, report = self.label([a, b])
        self.assertEqual(labels[24], 'H')
        self.assertTrue(any(c['kind'] == 'same_receiver_bridge' for c in report['claims']))
        labels, _ = self.label([a, event(2, kind='CHALLENGE', start=25, stop=25), b])
        self.assertEqual(labels[24], 'X')
        labels, _ = self.label([a, event(1, start=100, stop=110, sender='p2')])
        self.assertEqual(labels[50], 'X')

    def test_dead_requires_explicit_out_and_confirmed_restart(self):
        out = event(0, kind='BALL OUT', start=10, stop=20, receiver='')
        restart = event(1, kind='SET PIECE', start=80, stop=80, receiver='')
        labels, _ = self.label([out, restart])
        self.assertFalse(np.any(labels == 'D'))
        labels, _ = self.label([out, restart, event(2, start=80, stop=90)])
        self.assertTrue(np.all(labels[21:77] == 'D'))
        labels, _ = self.label([out, restart, event(2, start=80, stop=90), event(3, start=50, stop=60)])
        self.assertFalse(np.any(labels == 'D'))

    def test_overlapping_opposite_claims_are_unknown_and_order_independent(self):
        a = event(0, start=10, stop=40)
        b = event(1, team=1, start=25, stop=50, sender='p3', receiver='p4')
        y, r = self.label([a, b]); reverse, _ = self.label([b, a])
        np.testing.assert_array_equal(y, reverse)
        self.assertTrue(np.all(y[24:39] == 'X'))
        self.assertEqual(r['conflict_frames'], 15)

    def test_periods_and_source_frame_counters_are_respected(self):
        labels, _ = partial_labels(np.arange(1000, 1101), 2,
            [event(0, start=1005, stop=1020, period=2), event(1, start=10, stop=20)], self.roster)
        self.assertEqual(set(np.flatnonzero(labels == 'H')+1000), set(range(1007, 1018)))
        with self.assertRaises(ValueError):
            self.label([event(0, start=300, stop=320)])

    def test_held_out_event_file_is_not_opened_without_explicit_final_permission(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'sealed'):
                load_events(Path(folder), 3)

    def test_epts_substitution_layout_maps_active_players_to_persistent_slots(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)/'data/Sample_Game_3'; root.mkdir(parents=True)
            parameters = ''.join(f'<ProviderParameter><Name>{name}</Name><Value>{value}</Value></ProviderParameter>'
                for name, value in [('first_half_start', 1), ('first_half_end', 2), ('second_half_start', 3), ('second_half_end', 4)])
            channels = ''.join(f'<PlayerChannel id="p{i}_{axis}" playerId="p{i}" channelId="{axis}"/>'
                               for i in (1, 2, 3) for axis in ('x', 'y'))
            def layout(first, last, home):
                refs = ''.join(f'<PlayerChannelRef playerChannelId="p{i}_{axis}"/>' for i in (home, 3) for axis in ('x', 'y'))
                return f'<DataFormatSpecification startFrame="{first}" endFrame="{last}">{refs}</DataFormatSpecification>'
            metadata = ('<main><Team id="home"/><Team id="away"/>'
                '<Player id="p1" teamId="home"/><Player id="p2" teamId="home"/><Player id="p3" teamId="away"/>'
                '<FieldSize><Width>105</Width><Height>68</Height></FieldSize>'+parameters+channels+layout(1, 2, 1)+layout(3, 4, 2)+'</main>')
            (root/'Sample_Game_3_metadata.xml').write_text(metadata)
            (root/'Sample_Game_3_tracking.txt').write_text('1:0.1,0.2;0.8,0.9:0.5,0.5\n2:0.1,0.2;0.8,0.9:0.5,0.5\n3:0.2,0.3;0.8,0.9:0.5,0.5\n4:0.2,0.3;0.8,0.9:0.5,0.5\n')
            halves, _ = load_tracking(Path(folder), 3)
            self.assertTrue(np.isnan(halves[0].players[:, 1]).all())
            self.assertTrue(np.isnan(halves[1].players[:, 0]).all())
            np.testing.assert_allclose(halves[1].players[:, 1], np.tile([21., 20.4], (2, 1)), atol=1e-5)
            self.assertEqual(halves[1].frames.tolist(), [3, 4])

    def test_csv_frame_seconds_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'tracking.csv'
            path.write_text(',,,Home,,\n,,,1,,\nPeriod,Frame,Time [s],Player1,,Ball,\n1,1,1.0,.1,.2,.5,.5\n')
            with self.assertRaisesRegex(ValueError, '25 Hz'):
                load_csv_side(path)


if __name__ == '__main__':
    unittest.main()
