import torch
import argparse
import os.path as osp

from torch_sparse import SparseTensor
import scipy.sparse as sci_sparse


def dense_split(B, nparts, dim = 1):
    if nparts == 1:
        return [B.contiguous()]
    ret = torch.chunk(B, nparts, dim)
    return [item.contiguous() for item in ret]

class SparseTensorCOO:
    def __init__(self, coo: SparseTensor, dtype=torch.int32, format=""):
        # TODO replace int() with quantization.
        self.raw = coo
        self.dtype = dtype
        self.sp_info_ptr = None
        self.result = None
        self.parts = [self.raw]
        self.dense_parts = 0
        self.csr = []
        self.coo = []
        self.hidden_size = 0
        self.nparts = 1
        self.format = format


    def build_coo(self):
        self.coo = []
        for item in self.parts:
            row, col, value = item.coo()
            index = torch.stack([row, col], dim=0)
            if value is None:
                value = torch.ones(item.nnz(), dtype=self.dtype, device=item.device())
            else:
                value = value.type(self.dtype)
            self.coo.append(torch.sparse_coo_tensor(index,
                                                    value,
                                                    item.sizes()).coalesce())

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
        self.dense_parts = B_parts
        self.max_B_parts_ncols = (hidden_size + B_parts - 1) / B_parts
        max_h_size = (hidden_size + B_parts - 1) // B_parts
        if (len(self.csr) != len(self.parts)):
            self.build_csr()
        row_indices = [item.crow_indices() for item in self.csr]
        col_indices = [item.col_indices() for item in self.csr]
        values = [item.values() for item in self.csr]
        nrows = [item.size(0) for item in self.csr]
        ncols = [item.size(1) for item in self.csr]
        h_size = [max_h_size] * B_parts
        if B_parts * max_h_size != hidden_size:
            h_size[B_parts - 1] = hidden_size - (B_parts - 1) * max_h_size
        self.sp_info_ptr = torch.ops.pim_ops.spmm_csr_to_device_group(row_indices,
                                                                         col_indices,
                                                                         values,
                                                                         nrows,
                                                                         ncols,
                                                                         h_size,
                                                                         hidden_size)

    def to_pim_group_coo(self, hidden_size, B_parts = 4):
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
        self.sp_info_ptr = torch.ops.pim_ops.spmm_coo_to_device_group(self.row_indices,
                                                                      self.col_indices,
                                                                      self.values,
                                                                      nrows,
                                                                      ncols,
                                                                      h_size,
                                                                      hidden_size)

    def to_pim_group(self, hidden_size, B_parts=4):
        if self.format == "COO":
            self.to_pim_group_coo(hidden_size, B_parts)
        elif self.format == "CSR":
            self.to_pim_group_csr(hidden_size, B_parts)
        else:
            assert False

    def mul(self, B:torch.Tensor):
        assert self.hidden_size == B.size(1)
        B_parts = dense_split(B, self.dense_parts)
        if self.format == "CSR":
            res = torch.ops.pim_ops.spmm_csr_run_group(self.sp_info_ptr, B_parts)
        elif self.format == "COO":
            res = torch.ops.pim_ops.spmm_coo_run_group(self.sp_info_ptr, B_parts)
        else:
            res = None
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




TORCH_TYPES = {"INT64": torch.int64, "INT32": torch.int32, "INT16": torch.int16, "INT8": torch.int8, "FLT32":torch.float32, "DBL64":torch.float64}

def prepare_pim_spmm(adj_t: SparseTensor, args):
    A = SparseTensorCOO(adj_t, dtype=args.data_type, format=args.sp_format)
    A.col_split(args.sp_parts)
    A.to_pim_group(args.hidden_size, args.ds_parts)
    return A

def pim_spmm(x, adj_t: SparseTensorCOO):
    return adj_t.mul(x)

