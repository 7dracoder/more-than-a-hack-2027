"""Optional epoch-level learning-rate schedules with explicit resume state."""
import math
from torch.optim.lr_scheduler import CosineAnnealingLR


def make_scheduler(optimizer,config):
    kind=config.get('learning_rate_schedule','constant')
    if kind=='constant':
        return None
    if kind!='cosine':
        raise ValueError('Unknown learning-rate schedule: '+str(kind))
    epochs=config['epochs'];floor=config.get('minimum_learning_rate',1e-5)
    if not isinstance(epochs,int) or epochs<1:
        raise ValueError('Cosine scheduling needs a positive integer epoch count')
    if not math.isfinite(floor) or floor<0 or floor>min(group['lr'] for group in optimizer.param_groups):
        raise ValueError('Minimum learning rate must lie between zero and the initial rates')
    return CosineAnnealingLR(optimizer,T_max=epochs,eta_min=floor)
