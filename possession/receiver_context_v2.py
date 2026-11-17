"""Roster-padding-invariant normalization of the frozen receiver-context prototype."""
import numpy as np
from possession.receiver_context import actor_geometry,NAMES
from possession.receiver_context import marginal_context as original_context


def marginal_context(history,teams,ball_history,weights,pitch=(105.,68.),fps=25.):
    result=original_context(history,teams,ball_history,weights,pitch,fps)
    if result[-1]>0:
        # Never-observed slots have zero receiver probability. They must not
        # enlarge the entropy normalizer. Recently seen, stale candidates remain.
        recent=int(np.isfinite(np.asarray(history)).all(-1).any(0).sum())
        result[5]*=np.log(len(teams)+1)/np.log(max(2,recent+1))
    return result
