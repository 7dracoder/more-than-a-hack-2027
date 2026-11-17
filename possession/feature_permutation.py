"""Team exchange that preserves each occurrence of repeated feature names."""
from collections import Counter
import numpy as np


def team_swap_indices(names):
    used = Counter()
    keys = []
    for name in names:
        keys.append((name,used[name]))
        used[name] += 1
    positions = {key:i for i,key in enumerate(keys)}
    indices = []
    for name,occurrence in keys:
        other = 'away_'+name[5:] if name.startswith('home_') else 'home_'+name[5:] if name.startswith('away_') else name
        if (other,occurrence) not in positions:
            raise ValueError('Team feature occurrence has no matching counterpart')
        indices.append(positions[other,occurrence])
    result = np.asarray(indices,dtype=np.int64)
    if sorted(result.tolist()) != list(range(len(names))) or not np.array_equal(result[result],np.arange(len(names))):
        raise ValueError('Team exchange must be a bijective involution')
    return result
