"""Public Metrica adapters: XY features and partial event-derived targets stay separate."""
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from possession.external import TrackingHalf
from possession.data import slot_teams


@dataclass(frozen=True)
class Event:
    index: int
    period: int
    team: int
    kind: str
    subtype: str
    start: int
    stop: int
    sender: str
    receiver: str
    xy: tuple


def file_paths(root, number):
    directory = Path(root)/'data'/f'Sample_Game_{number}'
    prefix = directory/f'Sample_Game_{number}'
    if number in (1, 2):
        return dict(home=Path(str(prefix)+'_RawTrackingData_Home_Team.csv'),
                    away=Path(str(prefix)+'_RawTrackingData_Away_Team.csv'),
                    events=Path(str(prefix)+'_RawEventsData.csv'))
    if number == 3:
        return dict(tracking=Path(str(prefix)+'_tracking.txt'), metadata=Path(str(prefix)+'_metadata.xml'),
                    events=Path(str(prefix)+'_events.json'))
    raise ValueError('Expected public sample game 1, 2 or 3')


def load_csv_side(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.reader(stream); next(reader); next(reader); header = next(reader)
    data = pd.read_csv(path, header=None, skiprows=3, dtype=np.float32).to_numpy()
    roster = header[3:-2:2]
    if data.shape[1] != 3+2*len(roster)+2 or any(not p.startswith('Player') for p in roster):
        raise ValueError('Unexpected public CSV tracking schema')
    frame = data[:, 1].astype(np.int64); period = data[:, 0].astype(np.int8)
    if not np.array_equal(frame, data[:, 1]) or not np.isin(period, (1, 2)).all() or np.any(np.diff(frame) != 1):
        raise ValueError('CSV frame/period alignment changed')
    if np.max(np.abs(frame/25.-data[:, 2])) > .002:
        raise ValueError('Frame/time units are not the documented 25 Hz')
    return frame, period, data[:, 3:-2].reshape(len(data), len(roster), 2), data[:, -2:], roster


def load_tracking(root, number):
    """Never opens event files; timestamps and player names do not enter features."""
    paths = file_paths(root, number); pitch = (105., 68.)
    if number in (1, 2):
        f, period, home, ball, hnames = load_csv_side(paths['home'])
        af, ap, away, aball, anames = load_csv_side(paths['away'])
        if not np.array_equal(f, af) or not np.array_equal(period, ap):
            raise ValueError('Home and away tracking timelines differ')
        both = np.isfinite(ball).all(1) & np.isfinite(aball).all(1)
        if not np.allclose(ball[both], aball[both], rtol=0, atol=1e-6):
            raise ValueError('Duplicated ball tracks disagree')
        ball = np.where(np.isfinite(ball).all(1)[:, None], ball, aball)
        players = np.concatenate([home, away], axis=1)
        teams = np.r_[np.zeros(len(hnames), np.int8), np.ones(len(anames), np.int8)]
        roster = {name: 0 for name in hnames} | {name: 1 for name in anames}
    else:
        meta = ET.parse(paths['metadata']).getroot(); teams = slot_teams(paths['metadata'])
        field = next(meta.iter('FieldSize'))
        pitch = (float(field.findtext('Width')), float(field.findtext('Height')))
        root_teams = list(meta.iter('Team'))
        ids = {t.get('id'): i for i, t in enumerate(root_teams)}
        roster = {p.get('id'): ids[p.get('teamId')] for p in meta.iter('Player')}
        parameters = {p.findtext('Name'): p.findtext('Value') for p in meta.iter('ProviderParameter')}
        bounds = [(i, int(parameters[key+'_start']), int(parameters[key+'_end']))
                  for i, key in ((1, 'first_half'), (2, 'second_half'))]
        n = max(b for _, _, b in bounds)
        player_slot = {p.get('id'): i for i, p in enumerate(meta.iter('Player'))}
        channels = {c.get('id'): (player_slot[c.get('playerId')], {'x': 0, 'y': 1}[c.get('channelId')])
                    for c in meta.iter('PlayerChannel')}
        specifications = []; layout = np.full(n, -1, np.int16)
        for specification in meta.iter('DataFormatSpecification'):
            a, b = int(specification.get('startFrame')), int(specification.get('endFrame'))
            mapping = [channels[c.get('playerChannelId')] for c in specification.iter('PlayerChannelRef')]
            if len(set(mapping)) != len(mapping) or not 1 <= a <= b <= n or np.any(layout[a-1:b] >= 0):
                raise ValueError('Ambiguous EPTS channel layout')
            layout[a-1:b] = len(specifications); specifications.append(np.asarray(mapping, np.int64))
        if np.any(layout < 0):
            raise ValueError('EPTS channel layout has uncovered frames')
        players = np.full((n, len(teams), 2), np.nan, np.float32); ball = np.full((n, 2), np.nan, np.float32)
        f = np.arange(1, n+1); period = np.zeros(n, np.int8)
        for i, a, b in bounds:
            period[a-1:b] = i
        count = 0
        with paths['tracking'].open() as stream:
            for j, line in enumerate(stream):
                fields = line.strip().split(':')
                if len(fields) != 3 or int(fields[0]) != j+1 or j >= n:
                    raise ValueError('EPTS tracking timeline changed')
                row = np.fromstring(fields[1].replace(';', ','), sep=',', dtype=np.float32)
                point = np.fromstring(fields[2], sep=',', dtype=np.float32)
                mapping = specifications[layout[j]]
                if row.size != len(mapping) or point.size < 2:
                    raise ValueError('EPTS coordinate schema differs')
                players[j, mapping[:, 0], mapping[:, 1]] = row; ball[j] = point[:2]; count += 1
        if count != n or not np.isin(period, (1, 2)).all():
            raise ValueError('EPTS periods/tracking are incomplete')
    players *= np.array(pitch, np.float32); ball *= np.array(pitch, np.float32)
    halves = []
    for i in (1, 2):
        select = np.flatnonzero(period == i)
        if not len(select) or np.any(np.diff(select) != 1):
            raise ValueError('Each source half must be contiguous')
        halves.append(TrackingHalf('METRICA'+str(number), str(i), f[select], players[select], teams,
                                   ball[select], pitch, np.full(len(select), 'X', dtype='<U1')))
    return halves, roster


def load_events(root, number, allow_held_out=False):
    if number == 3 and not allow_held_out:
        raise ValueError('METRICA3 targets are sealed until model selection is frozen')
    path = file_paths(root, number)['events']; result = []
    if number in (1, 2):
        with path.open(encoding='utf-8-sig', newline='') as stream:
            for i, row in enumerate(csv.DictReader(stream)):
                result.append(Event(i, int(row['Period']), {'Home': 0, 'Away': 1}[row['Team']], row['Type'],
                    row['Subtype'], int(row['Start Frame']), int(row['End Frame']), row['From'], row['To'],
                    (float(row['Start X']), float(row['Start Y']))))
    else:
        meta = ET.parse(file_paths(root, number)['metadata']).getroot()
        sides = {t.get('id'): i for i, t in enumerate(meta.iter('Team'))}
        for row in json.loads(path.read_text(encoding='utf-8'))['data']:
            subtype = row.get('subtypes') or {}
            if isinstance(subtype, list):
                subtype = '-'.join(s.get('name', '') for s in subtype)
            else:
                subtype = subtype.get('name', '')
            start = row['start']; stop = row.get('end') or start
            result.append(Event(int(row['index']), int(row['period']), sides[row['team']['id']],
                row['type']['name'], subtype, int(start['frame']), int(stop['frame']),
                (row.get('from') or {}).get('id', ''), (row.get('to') or {}).get('id', ''),
                (float(start['x']) if start.get('x') is not None else np.nan,
                 float(start['y']) if start.get('y') is not None else np.nan)))
    if any(e.period not in (1, 2) or e.team not in (0, 1) for e in result):
        raise ValueError('Invalid event period/team')
    return sorted(result, key=lambda e: (e.period, e.start, e.index))


def controlled(event, roster):
    if event.kind not in ('PASS', 'CARRY') or any(s in event.subtype for s in ('HEAD', 'OFFSIDE', 'CLEARANCE', 'REFEREE')):
        return False
    if roster.get(event.sender) != event.team or not 0 < event.stop-event.start <= 250:
        return False
    return event.kind == 'CARRY' or roster.get(event.receiver) == event.team


def partial_labels(frames, period, events, roster, margin=2, maximum_bridge=50):
    """Only supported pass/carry/bridge or explicit-out-to-confirmed-restart intervals."""
    frames = np.asarray(frames)
    if not len(frames) or np.any(np.diff(frames) != 1) or margin < 0:
        raise ValueError('Expected one contiguous half and nonnegative boundary margin')
    current = [e for e in events if e.period == period]
    if any(e.start < frames[0] or e.start > frames[-1] or e.stop > frames[-1]+1 for e in current):
        raise ValueError('Event frames fall outside their documented half')
    good = [e for e in current if controlled(e, roster)]
    claims = np.zeros((len(frames), 3), bool); provenance = []
    def add(a, b, label, kind, indices):
        a, b = max(a, int(frames[0])), min(b, int(frames[-1])+1)
        if a < b:
            claims[a-frames[0]:b-frames[0], label] = True
            provenance.append(dict(start=int(a), stop=int(b), label='HAD'[label], kind=kind, events=indices))
    for e in good:
        add(e.start, e.stop, e.team, e.kind.lower(), [e.index])
    hazards = {'BALL LOST', 'BALL OUT', 'RECOVERY', 'CHALLENGE', 'FAULT RECEIVED', 'SHOT', 'SET PIECE'}
    for a, b in zip(good[:-1], good[1:]):
        if (a.team == b.team and a.receiver and a.receiver == b.sender and
                0 <= b.start-a.stop <= maximum_bridge and
                not any(e.kind in hazards and a.stop <= e.start <= b.start for e in current)):
            add(a.stop, b.start, a.team, 'same_receiver_bridge', [a.index, b.index])
    restarts = [e for e in current if e.kind == 'SET PIECE' and 'RETAKEN' not in e.subtype and
                any(p.start == e.start and p.team == e.team for p in good)]
    for e in current:
        if e.kind != 'BALL OUT' or e.stop < e.start:
            continue
        following = [r for r in restarts if r.start > e.stop]
        if not following:
            continue
        restart = following[0]
        if (restart.start-e.stop <= 3000 and
                not any(p.kind in ('PASS', 'CARRY', 'RECOVERY', 'SHOT') and e.stop < p.start < restart.start for p in current)):
            add(e.stop, restart.start, 2, 'explicit_out_to_confirmed_restart', [e.index, restart.index])
    labels = np.full(len(frames), 'X', dtype='<U1')
    for label in range(3):
        labels[(claims.sum(1) == 1) & claims[:, label]] = 'HAD'[label]
    before = labels.copy()
    edges = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1])+1, len(labels)]
    for a, b in zip(edges[:-1], edges[1:]):
        if labels[a] != 'X':
            labels[a:min(b, a+margin)] = 'X'; labels[max(a, b-margin):b] = 'X'
    return labels, dict(claims=provenance, conflict_frames=int((claims.sum(1) > 1).sum()),
                        boundary_frames_masked=int(((before != 'X') & (labels == 'X')).sum()),
                        label_counts={s: int((labels == s).sum()) for s in 'HADX'},
                        policy='Event-derived partial supervision; no labels inferred from event absence; no complete-spell truth claim.')
