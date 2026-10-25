"""Coordinate-only adapters and conservative H/A weak targets for public SkillCorner data."""
import json
import numpy as np


def weak_target(possession):
    group=(possession or {}).get('group')
    if group is None:return 'X'
    if group=='home team':return 'H'
    if group=='away team':return 'A'
    raise ValueError('Unknown provider possession group; do not infer a dead-ball label')


def roster(metadata):
    home,away=metadata['home_team']['id'],metadata['away_team']['id']
    ids=[];teams=[]
    for player in metadata['players']:
        if player['team_id'] not in (home,away):raise ValueError('Unknown team')
        ids.append(player['id']);teams.append(int(player['team_id']==away))
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate player identity')
    return ids,np.array(teams,np.int8)


def coordinates(row,index,detected_only=False):
    players=np.full((len(index),2),np.nan,np.float32);seen=set()
    for player in row.get('player_data',[]):
        identity=player['player_id']
        if identity not in index or identity in seen:raise ValueError('Unknown or duplicate tracked player')
        seen.add(identity)
        if not detected_only or player.get('is_detected') is True:
            players[index[identity]]=[np.nan if player.get(k) is None else player[k] for k in ('x','y')]
    ball_data=row.get('ball_data') or {}
    ball=np.full(2,np.nan,np.float32)
    if not detected_only or ball_data.get('is_detected') is True:
        ball[:]=[np.nan if ball_data.get(k) is None else ball_data[k] for k in ('x','y')]
    players[~np.isfinite(players)]=np.nan;ball[~np.isfinite(ball)]=np.nan
    return players,ball


def resample_coordinates(values,source_fps=10.,target_fps=25.,speed_limit=15.):
    """Interpolate only adjacent finite source observations without teleporting."""
    values=np.asarray(values,dtype=np.float32)
    if not len(values) or values.shape[-1]!=2 or min(source_fps,target_fps,speed_limit)<=0:
        raise ValueError('Invalid coordinate timeline')
    ticks=np.arange(int(np.floor((len(values)-1)*target_fps/source_fps))+1,dtype=np.float64)*source_fps/target_fps
    nearest=np.rint(ticks).astype(np.int64);exact=np.isclose(ticks,nearest,atol=1e-9,rtol=0)
    low=np.floor(ticks).astype(np.int64);high=np.minimum(low+1,len(values)-1)
    low[exact]=nearest[exact];high[exact]=nearest[exact]
    left,right=values[low],values[high]
    valid=np.isfinite(left).all(-1)&np.isfinite(right).all(-1)
    with np.errstate(invalid='ignore'):
        speed=np.linalg.norm(right-left,axis=-1)*source_fps
        valid&=speed<=speed_limit
        fraction=(ticks-low).reshape((-1,)+(1,)*(values.ndim-1))
        result=left+(right-left)*fraction
    return np.where(valid[...,None],result,np.nan).astype(np.float32),low


def load_match(metadata,path):
    ids,teams=roster(metadata);index={identity:i for i,identity in enumerate(ids)}
    halves={}
    for period in metadata['match_periods']:
        if period['period'] not in (1,2):continue
        first,last=int(period['start_frame']),int(period['end_frame']);length=last-first+1
        if length<=0:raise ValueError('Invalid period bounds')
        halves[period['period']]=dict(first=first,last=last,labels=np.full(length,'X',dtype='<U1'),
            native=(np.full((length,len(ids),2),np.nan,np.float32),np.full((length,2),np.nan,np.float32)),
            detected=(np.full((length,len(ids),2),np.nan,np.float32),np.full((length,2),np.nan,np.float32)),seen=np.zeros(length,bool))
    previous=-1
    with open(path) as stream:
        for line in stream:
            row=json.loads(line);frame=row['frame']
            if frame<=previous:raise ValueError('Tracking frame order is not strictly increasing')
            previous=frame;period=row.get('period')
            if period not in halves:continue
            half=halves[period];tick=frame-half['first']
            if not 0<=tick<len(half['labels']):raise ValueError('Tracking period and metadata bounds disagree')
            half['seen'][tick]=True;half['labels'][tick]=weak_target(row.get('possession'))
            for view in ('native','detected'):
                players,ball=coordinates(row,index,view=='detected')
                half[view][0][tick]=players;half[view][1][tick]=ball
    if set(halves)!={1,2}:raise ValueError('Two regular periods are required')
    return halves,teams
