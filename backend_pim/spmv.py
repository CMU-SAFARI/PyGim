import torch
import argparse
import os.path as osp
import torch_geometric.transforms as T
import torch_geometric.utils as pygutils
from torch_geometric.datasets import Planetoid, Reddit
from torch_geometric import datasets
from ogb.nodeproppred import PygNodePropPredDataset
from ogb.linkproppred import PygLinkPropPredDataset
from ogb.graphproppred import PygGraphPropPredDataset
from torch_sparse import SparseTensor
import scipy.sparse as sci_sparse
import torch_geometric.utils as pyg_utils

def dense_split(B, nparts, dim = 1):
    if nparts == 1:
        return [B.contiguous()]
    ret = torch.chunk(B, nparts, dim)
    return [item.contiguous() for item in ret]

class SparseTensorCOO:
    def __init__(self, coo: SparseTensor, dtype=torch.int32, groups=32):
        # TODO replace int() with quantization.
        self.raw = coo.int()
        self.dtype = dtype
        self.sp_info_ptr = None
        self.result = None
        self.parts = [self.raw]
        self.dense_parts = 0
        self.csr = []
        self.coo = []
        self.hidden_size = 0
        self.nparts = 1
        self.format = ""
        self.groups = groups

    def __del__(self):
        pass
        # TODO without prepare_pim(), free will cause error.
        # if not(self.sp_info_ptr is None):
        #     torch.ops.pim_ops.spmm_free_group(self.sp_info_ptr)

    def build_coo(self):
        self.coo = []
        min_length = 64 // torch.iinfo(self.dtype).bits
        for item in self.parts:
            pad_size = 0
            if (item.size(0) % min_length != 0):
                pad_size = min_length - item.size(0) % min_length
            nrows = item.size(0) + pad_size
            ncols = item.size(1) + pad_size
            row, col, value = item.coo()
            index = torch.stack([row, col], dim=0)
            if value is None:
                value = torch.ones(item.nnz(), dtype=self.dtype, device=item.device())
            else:
                value = value.type(self.dtype)
            self.coo.append(torch.sparse_coo_tensor(index,
                                                    value,
                                                    (nrows, ncols)).coalesce())


    def to_pim_group_coo(self, hidden_size, rank_pre_spmv = 1):
        B_parts = hidden_size
        self.format = "COO"
        self.hidden_size = hidden_size
        self.dense_parts = B_parts
        self.max_B_parts_ncols = (hidden_size + B_parts - 1) / B_parts
        max_h_size = (hidden_size + B_parts - 1) // B_parts
        if (len(self.coo) != len(self.parts)):
            self.build_coo()
        self.row_indices = [item.indices()[0].int().contiguous() for item in self.coo]
        self.col_indices = [item.indices()[1].int().contiguous() for item in self.coo]
        self.values = [item.values() for item in self.coo]
        nrows = [item.size(0) for item in self.coo]
        ncols = [item.size(1) for item in self.coo]
        h_size = [max_h_size] * B_parts
        if B_parts * max_h_size != hidden_size:
            h_size[B_parts - 1] = hidden_size - (B_parts - 1) * max_h_size
        self.sp_info_ptr = torch.ops.pim_ops.spmv_coo_to_device_group(self.row_indices,
                                                                      self.col_indices,
                                                                      self.values,
                                                                      nrows,
                                                                      ncols,
                                                                      h_size,
                                                                      hidden_size, rank_pre_spmv)


    def mul_single(self, B:torch.Tensor):
        assert self.hidden_size == B.size(1)
        B_parts = dense_split(B, self.dense_parts)
        res = torch.ops.pim_ops.spmv_coo_run_group(self.sp_info_ptr, B_parts)
        return res[:self.raw.size(0), ...]

    def mul(self, B:torch.Tensor):
        Bs = dense_split(B, B.size(1) // self.groups)
        res = []
        for item in Bs:
            res_tmp = self.mul_single(item)
            res.append(res_tmp)
        res_lib = torch.cat(res, dim=1)
        return res_lib

    def row_split(self, nparts = 4):
        assert False

    def col_split(self, nparts = 4):
        assert False


TORCH_TYPES = {"INT64": torch.int64, "INT32": torch.int32, "INT16": torch.int16, "INT8": torch.int8, "FLT32":torch.float32, "DBL64":torch.float64}

def prepare_pim_spmv(adj_t: SparseTensor, args):
    assert args.sp_format == "COO"
    A = SparseTensorCOO(adj_t, dtype=args.data_type, groups=args.ds_parts)
    A.to_pim_group_coo(args.ds_parts, args.sp_parts)
    return A

def pim_spmv(x, adj_t: SparseTensorCOO):
    return adj_t.mul(x)