"""Choose native versus corrupted external training views explicitly."""
import math


def validate_view_sampling(total,native,probability):
    if probability is None:return
    if not isinstance(total,int) or not isinstance(native,int) or native<=0 or total!=2*native:
        raise ValueError('View weighting requires one native and one augmented bank')
    if not isinstance(probability,(int,float)) or not math.isfinite(probability) or not 0<=probability<=1:
        raise ValueError('Native training view probability must lie between zero and one')


def sample_training_view(rng,total,native,probability=None):
    # Preserve the original random stream exactly for all existing configs.
    if probability is None:return int(rng.integers(total))
    validate_view_sampling(total,native,probability)
    offset=0 if rng.random()<probability else native
    return offset+int(rng.integers(native))
