"""Development-fitted common evidence scores and preserving detector unions."""
import numpy as np
from possession.tiled_ball_detection import append_nonoverlapping


def evidence_scores(confidence,parameters):
    from scipy.special import expit
    confidence=np.clip(np.asarray(confidence,float),1e-6,1-1e-6)
    return expit(parameters['slope']*np.log(confidence/(1-confidence))+parameters['intercept'])


def compose(branches,parameters,kind,tile_threshold=None):
    def selected(name,threshold):
        row=branches[name];keep=row['confidence']>=threshold
        return row['boxes'][keep].copy(),evidence_scores(row['confidence'][keep],parameters[name])
    if kind=='yolo':return selected('yolo11m',.5)
    if kind=='rfdetr':return selected('rfdetr_large',.8)
    if kind not in ('union','union_tiled'):raise ValueError('Unknown frozen detector composition')
    boxes,scores=selected('yolo11m',.5);new_boxes,new_scores=selected('rfdetr_large',.8)
    boxes,scores=append_nonoverlapping(boxes,scores,new_boxes,new_scores,.5)
    if kind=='union_tiled':
        if tile_threshold is None:raise ValueError('A frozen tile threshold is required')
        new_boxes,new_scores=selected('rfdetr_tiled',tile_threshold)
        boxes,scores=append_nonoverlapping(boxes,scores,new_boxes,new_scores,.5)
    if len(boxes)>300:raise ValueError('Frozen comparison exceeded its maximum detection count')
    return boxes,scores
