"""Keep explicitly declared observation masks in their original zero/one units."""
import numpy as np


def preserve_unit_masks(mean,scale,names,unit_names):
    if not unit_names:return mean,scale
    if len(mean)!=len(names) or np.shape(mean)!=np.shape(scale):
        raise ValueError('Normalization schema differs')
    if len(unit_names)!=len(set(unit_names)) or any(n not in names or not n.endswith('_known') for n in unit_names):
        raise ValueError('Unit masks must be unique known-feature names in the schema')
    mean=mean.copy();scale=scale.copy()
    index=[names.index(n) for n in unit_names]
    mean[index]=0.;scale[index]=1.
    return mean,scale


def preserve_trajectory_units(mean,scale,names):
    """Use the producer's shared physical units for every player trajectory slot.

    Per-rank centering would give the shared encoder different representations
    of the same displacement. The producer already scales metres by ten.
    """
    from possession.features_v25 import OFFSETS,COMPONENTS
    expected=[f'{side}_tube{rank}_o{offset}_{component}'
              for side in ('home','away') for rank in (1,2,3)
              for offset in OFFSETS for component in COMPONENTS]
    if len(names)!=960 or list(names[-180:])!=expected or np.shape(mean)!=(960,) or np.shape(scale)!=(960,):
        raise ValueError('Physical trajectory units require the exact 780+180 schema')
    mean=mean.copy();scale=scale.copy()
    mean[-180:]=0.;scale[-180:]=1.
    return mean,scale
