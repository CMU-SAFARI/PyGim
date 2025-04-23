from typing import Callable, Optional, Union

import torch
from torch import Tensor
from torch_sparse import SparseTensor, matmul


from torch_sparse import SparseTensor



def spmm_cpu_int(A, B):
    row_indices = A.crow_indices().int()
    col_indices = A.col_indices().int()
    vaules = A.values().int()
    B = B.int()
    res = torch.ops.pim_ops.spmm_cpu(row_indices, col_indices, vaules, B)
    return res

def symmetric_quantize(v, dtype = torch.int32):
    abs_max = torch.max(v.abs())
    if (dtype == torch.int8):
        scale = abs_max * 2 / pow(2, 5)
    elif (dtype == torch.int16):
        scale = abs_max * 2 / pow(2, 10)
    elif (dtype == torch.int32):
        scale = abs_max * 2 / pow(2, 20)
    else:
        scale = abs_max*2/pow(2, 20)
        dtype = torch.float
    new_v = torch.round(v/scale)

    if (new_v.dtype == dtype):
        new_v = new_v.clone()
    else:
        new_v = new_v.to(dtype)
    # new_v = torch.round(v/scale)
    return scale, new_v

def symmetric_dequantize(out, scale_edge, scale_x):
    out = out * (scale_edge * scale_x)
    return out