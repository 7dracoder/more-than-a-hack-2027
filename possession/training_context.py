"""Ignore artificial crop padding when fitting a model with finite context."""
def mask_crop_context(labels,weights,radius):
    if labels.ndim!=2 or labels.shape!=weights.shape:
        raise ValueError('Training labels and weights must share a batch/time shape')
    if not isinstance(radius,int) or radius<0 or 2*radius>=labels.shape[1]:
        raise ValueError('Training crop must contain an interior beyond both context margins')
    if radius:
        labels[:,:radius]=-100;labels[:,-radius:]=-100
        weights[:,:radius]=0.;weights[:,-radius:]=0.
