"""Additional coordinate-only evidence for brief touches and ball direction changes."""
import warnings
import numpy as np
from possession.features_v08 import build_features as base_features, velocity
from possession.external import finite_mean


def shift(values, offset):
    result = np.full_like(values, np.nan)
    if offset > 0:
        result[offset:] = values[:-offset]
    elif offset < 0:
        result[:offset] = values[-offset:]
    else:
        result[:] = values
    return result


def build_features(players, teams, ball, pitch, fps=25., step=2):
    base, names = base_features(players, teams, ball, pitch, fps=fps, step=step)
    players, ball, teams = np.asarray(players), np.asarray(ball), np.asarray(teams)
    bv, pv = velocity(ball, fps), velocity(players, fps, max_speed=15.)
    acceleration = velocity(bv, fps, max_speed=500.)
    columns = dict(ball_acceleration=np.linalg.norm(acceleration, axis=-1).clip(0,500)/100.)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for lag in (2,5,10):
            before, after = shift(bv, lag), shift(bv, -lag)
            denominator = np.linalg.norm(before,axis=-1)*np.linalg.norm(after,axis=-1)
            dot = np.sum(before*after,axis=-1)
            cosine = np.divide(dot,denominator,out=np.full(len(ball),np.nan),where=denominator>.1)
            columns[f'ball_turn{lag}'] = 1-np.clip(cosine,-1,1)
        for side, prefix in [(0,'home'),(1,'away')]:
            xy, speed = players[:,teams==side], pv[:,teams==side]
            delta = xy-ball[:,None,:]
            distance = np.linalg.norm(delta,axis=-1)
            relative = speed-bv[:,None,:]
            relative_speed = np.linalg.norm(relative,axis=-1)
            for horizon in (-.2,.2,.5):
                projected = np.linalg.norm(delta+relative*horizon,axis=-1)
                minimum = np.min(np.where(np.isfinite(projected),projected,np.inf),axis=1)
                columns[f'{prefix}_projected_distance{horizon:g}'] = np.where(np.isfinite(minimum),minimum.clip(0,30)/10.,np.nan)
            for radius in (1.,3.,5.):
                weight = np.where(np.isfinite(relative_speed),np.exp(-distance**2/(2*radius**2)),0.)
                mass = weight.sum(1)
                def average(value):
                    actual_weight = np.where(np.isfinite(value),weight,0.)
                    actual_mass = actual_weight.sum(1)
                    return np.divide((actual_weight*np.where(np.isfinite(value),value,0.)).sum(1),actual_mass,
                                     out=np.full(len(ball),np.nan),where=actual_mass>1e-5)
                columns[f'{prefix}_relative_motion{radius:g}'] = average(relative_speed).clip(0,60)/20.
                similarity = np.sum(speed*bv[:,None,:],axis=-1)
                denominator = np.linalg.norm(speed,axis=-1)*np.linalg.norm(bv,axis=-1)[:,None]
                cosine = np.divide(similarity,denominator,out=np.full_like(similarity,np.nan),where=denominator>.1)
                columns[f'{prefix}_velocity_alignment{radius:g}'] = average(cosine)
                columns[f'{prefix}_motion_mass{radius:g}'] = mass/3.
            minimum = np.min(np.where((distance<3.)&np.isfinite(relative_speed),relative_speed,np.inf),axis=1)
            columns[f'{prefix}_close_relative_minimum'] = np.where(np.isfinite(minimum),minimum.clip(0,60)/20.,np.nan)
    raw = list(columns.items())
    for key,value in raw:
        for radius in (3,10):
            columns[f'{key}_mean{radius}'] = finite_mean(value,radius)
    extra_names = list(columns)
    values = np.stack([columns[key][::step] for key in extra_names],axis=1)
    known = np.isfinite(values)
    extra = np.concatenate([np.where(known,values,0.),known.astype(np.float32)],axis=1)
    return np.concatenate([base,extra.astype(np.float32)],axis=1), names+extra_names+[name+'_known' for name in extra_names]
