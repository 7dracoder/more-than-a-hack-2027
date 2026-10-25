"""Ephemeral, checksum-guarded evaluation reuse for a completely frozen teacher."""
from dataclasses import dataclass
import hashlib
import json
import numpy as np
import torch


def input_digest(matrix,names,mean,scale):
    result=hashlib.sha256(json.dumps(list(names)).encode())
    for array in (matrix,mean,scale):
        array=np.asarray(array)
        result.update(str((array.shape,array.dtype.str)).encode())
        result.update(memoryview(np.ascontiguousarray(array)))
    return result.hexdigest()


def teacher_digest(model):
    if not hasattr(model,'teacher') or any(p.requires_grad for p in model.teacher.parameters()):
        raise ValueError('Evaluation reuse requires a completely frozen residual teacher')
    if any(module.training for module in model.teacher.modules()):
        raise ValueError('Frozen teacher must stay in evaluation mode')
    calibration=[(name,module.positive_weight) for name,module in model.teacher.named_modules() if hasattr(module,'positive_weight')]
    result=hashlib.sha256(str((model.teacher_feature_count,repr(model.teacher),calibration)).encode())
    for name,value in model.teacher.state_dict().items():
        array=value.detach().cpu().contiguous().numpy()
        result.update(name.encode());result.update(str((array.shape,array.dtype.str)).encode());result.update(memoryview(array))
    return result.hexdigest()


@dataclass(frozen=True)
class TeacherCache:
    input_sha256: str
    teacher_sha256: str
    state: np.ndarray
    edge: np.ndarray

    def outputs(self,model,matrix,names,mean,scale,device):
        if teacher_digest(model)!=self.teacher_sha256 or input_digest(matrix,names,mean,scale)!=self.input_sha256:
            raise ValueError('Cached teacher input, normalization or weights changed')
        return tuple(torch.as_tensor(value.copy(),device=device) for value in (self.state,self.edge))


def build_teacher_cache(model,matrix,names,mean,scale,device):
    from possession.temporal import normalized
    from possession.external import swap_features
    before=teacher_digest(model)
    # Exactly the same whole-half, original/swapped input as normal inference.
    original=normalized(matrix,mean,scale)
    swapped=normalized(swap_features(matrix,names),mean,scale)
    batch=torch.as_tensor(np.stack([original.T,swapped.T]),device=device)
    model.teacher.eval()
    with torch.inference_mode():
        state,edge=model.teacher.forward_with_boundary(batch[:,:model.teacher_feature_count])
        state=state.cpu().numpy().copy();edge=edge.cpu().numpy().copy()
    if before!=teacher_digest(model):raise ValueError('Teacher changed while caching')
    state.flags.writeable=False;edge.flags.writeable=False
    return TeacherCache(input_digest(matrix,names,mean,scale),before,state,edge)
