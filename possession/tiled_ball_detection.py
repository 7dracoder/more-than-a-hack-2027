"""Fixed crop geometry and conservative, geometry-preserving ball unions."""
import math
import numpy as np


def crop_windows(width,height):
    if not isinstance(width,int) or not isinstance(height,int) or min(width,height)<1:raise ValueError('Need positive integer image dimensions')
    w,h=math.ceil(.6*width),math.ceil(.6*height)
    return list(dict.fromkeys((x,y,x+w,y+h) for y in (0,height-h) for x in (0,width-w)))


def translate_boxes(boxes,window,width,height):
    boxes=np.asarray(boxes,float).reshape(-1,4).copy();x,y,_,_=window
    boxes+=np.array([x,y,x,y]);boxes[:,[0,2]]=boxes[:,[0,2]].clip(0,width);boxes[:,[1,3]]=boxes[:,[1,3]].clip(0,height)
    return boxes


def iou_one(box,others):
    others=np.asarray(others,float).reshape(-1,4)
    intersection=np.maximum(0,np.minimum(box[2:],others[:,2:])-np.maximum(box[:2],others[:,:2])).prod(1)
    union=np.maximum(0,box[2:]-box[:2]).prod()+np.maximum(0,others[:,2:]-others[:,:2]).prod(1)-intersection
    return intersection/np.maximum(union,1e-12)


def nms(boxes,scores,iou=.5,limit=300):
    boxes=np.asarray(boxes).reshape(-1,4);scores=np.asarray(scores)
    if scores.shape!=(len(boxes),) or not np.isfinite(boxes).all() or not np.isfinite(scores).all():raise ValueError('Invalid detections')
    order=np.argsort(-scores,kind='stable');retained=[]
    for i in order:
        if retained and np.any(iou_one(boxes[i],boxes[retained])>=iou):continue
        retained.append(int(i))
        if len(retained)==limit:break
    return boxes[retained],scores[retained]


def append_nonoverlapping(boxes,scores,new_boxes,new_scores,iou=.5):
    boxes=np.asarray(boxes).reshape(-1,4).copy();scores=np.asarray(scores).copy()
    new_boxes=np.asarray(new_boxes).reshape(-1,4);new_scores=np.asarray(new_scores)
    for index in np.argsort(-new_scores,kind='stable'):
        if len(boxes) and np.any(iou_one(new_boxes[index],boxes)>=iou):continue
        boxes=np.concatenate([boxes,new_boxes[index:index+1]]);scores=np.r_[scores,new_scores[index]]
    return boxes,scores
