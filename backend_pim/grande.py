import torch
import argparse
import os.path as osp
import datetime
from torch_sparse import SparseTensor
import scipy.sparse as sci_sparse



TORCH_TYPES = {"INT64": torch.int64, "INT32": torch.int32, "INT16": torch.int16, "INT8": torch.int8, "FLT32":torch.float32, "DBL64":torch.float64}
TYPES_MUL = {torch.int64: 1, torch.int32: 2, torch.int16: 4, torch.int8: 8, torch.float32: 2, torch.float64: 1}
def dense_split(B, ncols, dim = 1):
    pad_ncols = (ncols[0] + TYPES_MUL[B.dtype] - 1) // TYPES_MUL[B.dtype] * TYPES_MUL[B.dtype]
    if (ncols[-1] % pad_ncols != 0):
        B = torch.nn.functional.pad(B, (0, pad_ncols - ncols[-1] % pad_ncols))
    if len(ncols) == 1:
        return [B.contiguous()]
    ret = []
    cur_col = 0
    for ncol in ncols:
        ret.append(B[:, cur_col : cur_col+pad_ncols])
        cur_col += ncol
    return [item.contiguous() for item in ret]

class SparseTensorCOO:
    def __init__(self, coo: SparseTensor, dtype=torch.int32, dpus_per_rank = [], format=""):
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
        self.dpus_per_rank = dpus_per_rank
        self.format = ""

    def build_csr(self):
        self.csr = []
        for item in self.parts:
            rowptr, col, value = item.csr()
            if value is None:
                value = torch.ones(item.nnz(), dtype=self.dtype, device=item.device())
            else:
                value = value.type(self.dtype)
            self.csr.append(torch.sparse_csr_tensor(rowptr.int().contiguous(),
                                                    col.int().contiguous(),
                                                    value.contiguous(),
                                                    item.sizes()))

    def to_pim_group_csr(self, hidden_size, B_parts = 4):
        self.format = "CSR"
        self.hidden_size = hidden_size
        if (len(self.csr) != len(self.parts)):
            self.build_csr()
        row_indices = [item.crow_indices() for item in self.csr]
        col_indices = [item.col_indices() for item in self.csr]
        values = [item.values() for item in self.csr]
        nrows = [item.size(0) for item in self.csr]
        ncols = [item.size(1) for item in self.csr]
        h_size = []
        for i, item in enumerate(self.csr):
            B_parts = self.dpus_per_rank[i]
            # B_cols = (hidden_size + TYPES_MUL[self.dtype] - 1) // TYPES_MUL[self.dtype] * TYPES_MUL[self.dtype]
            rank_h_size = [hidden_size // B_parts] * B_parts
            rest = hidden_size - rank_h_size[0] * B_parts
            for i in range(rest):
                rank_h_size[i] += 1
            # print(rank_h_size)
            h_size.append(torch.tensor(rank_h_size, dtype=torch.int32))
        self.dense_ncols = h_size
        self.sp_info_ptr = torch.ops.pim_ops.spmm_csr_to_device_group(row_indices,
                                                                         col_indices,
                                                                         values,
                                                                         nrows,
                                                                         ncols,
                                                                         self.dense_ncols,
                                                                         hidden_size)


    def mul(self, B:torch.Tensor):
        assert self.hidden_size == B.size(1)
        assert len(self.dpus_per_rank) == len(self.csr)
        row_splits = [item.size(1) for item in self.csr]
        start_split = datetime.datetime.now()
        row_splits = torch.split(B, row_splits, dim=0)
        end_split_row = datetime.datetime.now()
        B_parts = []
        for i, item in enumerate(row_splits):
            B_parts += dense_split(item, self.dense_ncols[i])
        end_split_all = datetime.datetime.now()
        # print("[DATA]dense_split_row_time(ms): ", (end_split_row - start_split).total_seconds() * 1000, flush=True)
        # print("[DATA]dense_split_all_time(ms): ", (end_split_all - start_split).total_seconds() * 1000, flush=True)


        start_spmm_all = datetime.datetime.now()
        if self.format == "CSR":
            res = torch.ops.pim_ops.spmm_csr_run_group(self.sp_info_ptr, B_parts)
        elif self.format == "COO":
            res = torch.ops.pim_ops.spmm_coo_run_group(self.sp_info_ptr, B_parts)
        else:
            res = None
        end_spmm_all = datetime.datetime.now()
        # print("[DATA]mul_backend_time(ms): ", (end_spmm_all - start_spmm_all).total_seconds() * 1000, flush=True)
        return res

    def row_split(self, nparts = 4):
        assert False

    def col_split(self, nparts = 4):
        assert nparts > 0
        max_length = (self.raw.size(1) + nparts - 1) // nparts
        if (nparts != len(self.parts)):
            assert len(self.parts) == 1
            self.parts = [self.raw[:, i*max_length:(i+1)*max_length] for i in range(nparts - 1)]
            self.parts.append(self.raw[:, (nparts - 1)*max_length:])
            self.csr = []
            self.coo = []
        return self.parts


def prepare_pim_spmm_grande(adj_t: SparseTensor, args, dpus_per_rank):
    A = SparseTensorCOO(adj_t, dtype=args.data_type, dpus_per_rank = dpus_per_rank, format=args.sp_format)
    A.col_split(args.sp_parts)
    A.to_pim_group_csr(args.hidden_size)
    return A

def pim_spmm_grande(x, adj_t: SparseTensorCOO):
    return adj_t.mul(x)
