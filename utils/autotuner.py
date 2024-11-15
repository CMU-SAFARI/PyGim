import torch
import numpy as np
import argparse
import os.path as osp
import torch_geometric.transforms as T
import torch_geometric.utils as pygutils
from torch_geometric import datasets
from torch_geometric.datasets import Planetoid, Reddit
from ogb.nodeproppred import PygNodePropPredDataset
from ogb.linkproppred import PygLinkPropPredDataset
from ogb.graphproppred import PygGraphPropPredDataset
from torch_sparse import SparseTensor, matmul
import scipy.sparse as sci_sparse
import torch_geometric.utils as pyg_utils
import datetime


TORCH_TYPES = {"INT64": torch.int64, "INT32": torch.int32, "INT16": torch.int16, "INT8": torch.int8, "FLT32":torch.float32, "DBL64":torch.float64}
# TYPES_BYTES = {"INT64": 8, "INT32": 4, "INT16": 2, "INT8": 1, "FLT32": 4, "DBL64": 8}
TYPES_BYTES = {torch.int64: 8, torch.int32: 4, torch.int16: 2, torch.int8: 1, torch.float32: 4, torch.float64: 8}

# the unit is bytes/ops per 1e-3 s
DRAM_BANDWIDTH = 23.1*1e6
DPU_HOST_BANDWIDTH = {
    64 : 4.839043 * 1e6,
    96 : 4.96115 * 1e6,
    128 : 4.485867 * 1e6,
    160 : 4.993945 * 1e6,
    192 : 5.128233 * 1e6,
    224 : 5.349502 * 1e6,
    256 : 4.121898 * 1e6,
    288 : 5.763526 * 1e6,
    320 : 5.079638 * 1e6,
    352 : 5.871919 * 1e6,
    384 : 5.682741 * 1e6,
    416 : 5.873649 * 1e6,
    448 : 6.064049 * 1e6,
    480 : 6.354006 * 1e6,
}

HOST_DPU_BANDWIDTH = 43.73*1e6
# PIM_OPs = {torch.int32: 8.73763*1e3, torch.float32: 1.375*1e3}
PIM_OPs = {torch.int32: 29.09889*1e3, torch.float32: 6.237449*1e3}

# dense_size : nr_rows per ms
# (0+1) 0.14+0.17+0.26+0.48+0.87=1.92 sec
INT32_READ_FMA_THROUGHPUT = {2:3.52726 * 1e3,
                             4:3.03381 * 1e3,
                             8:1.76268 * 1e3,
                             16:0.96318 * 1e3,
                             32:0.50720 * 1e3,}
# dense_size : nr_rows per ms
# around 0.4 sec for each
DRAM_ROW_COPY_THROUGHPUT = { 2:28.41360667 * 1e3,
                             4:27.39013667 * 1e3,
                             8:23.20701 * 1e3,
                             16:17.06662667 * 1e3,
                             32:10.16865 * 1e3,}
DRAM_ROW_ADD_THROUGHPUT = {2:1e200,
                           4:36.84901667 * 1e3,
                           8:32.65140667 * 1e3,
                           16:24.90242 * 1e3,
                           32:15.50626667 * 1e3,}
# real data: 
# 4: 26.650877 (26.43862 25.75695 27.75706)
# 8: 23.30964667 (22.81343 22.77099 23.47864 23.27991 23.86274 23.65217 )
# 16: 17.16208833 (18.17138 16.42234 17.11752 17.79278 17.42661 16.04190)
# 32: 10.362397 (10.15231 10.55400 10.38088)

# Bytes : latency in 1e-6 s
READ_LATENCY = {8:0.1881544931,
                16:0.2116339547,
                32:0.2550986154,
                64:0.2600784302,
                128:0.3697618757,
                256:0.6018327985,
                512:0.9282749721,
                1024:1.664936175,
                2048:3.282903181,}

WRITE_LATENCY = {8:0.1369789668,
                 16:0.1548164913,
                 32:0.1519159589,
                 64:0.2059991019,
                 128:0.4119982038,
                 256:0.4721374512,
                 512:0.8861258371,
                 1024:1.628398786,
                 2048:3.268575614,}

def find_block_size(bytes):
    bytes /= 1024
    size = 64
    while (size + 32 < bytes):
        if size == 480:
            break
        else:
            size += 32
    if size < 480 and np.abs(bytes - size) > np.abs(size + 32 - bytes):
        size += 32
    return size

def dense_split(B, nparts, dim = 1):
    if nparts == 1:
        return [B.contiguous()]
    ret = torch.chunk(B, nparts, dim)
    return [item.contiguous() for item in ret]

class SparseTensorCOO_a:
    def __init__(self, coo: SparseTensor, dtype=torch.int32):
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

    def __del__(self):
        pass
        # TODO without prepare_pim(), free will cause error.
        # if not(self.sp_info_ptr is None):
        #     torch.ops.pim_ops.spmm_free_group(self.sp_info_ptr)

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
        self.max_h_size = (hidden_size + B_parts - 1) // B_parts
        if (len(self.csr) != len(self.parts)):
            self.build_csr()
        row_indices = [item.crow_indices() for item in self.csr]
        col_indices = [item.col_indices() for item in self.csr]
        values = [item.values() for item in self.csr]
        nrows = [item.size(0) for item in self.csr]
        ncols = [item.size(1) for item in self.csr]
        h_size = [self.max_h_size] * B_parts
        if B_parts * self.max_h_size != hidden_size:
            h_size[B_parts - 1] = hidden_size - (B_parts - 1) * self.max_h_size
        st = datetime.datetime.now()
        self.sp_info_ptr = torch.ops.pim_ops.spmm_csr_to_device_group(row_indices,
                                                                         col_indices,
                                                                         values,
                                                                         nrows,
                                                                         ncols,
                                                                         h_size,
                                                                         hidden_size)
        end = datetime.datetime.now()
        self.prepare_time = (end - st).total_seconds() * 1000
        

    def to_pim_group_coo(self, hidden_size, B_parts = 4):
        self.format = "COO"
        self.hidden_size = hidden_size
        self.dense_parts = B_parts
        self.max_B_parts_ncols = (hidden_size + B_parts - 1) / B_parts
        self.max_h_size = (hidden_size + B_parts - 1) // B_parts
        if (len(self.coo) != len(self.parts)):
            self.build_coo()
        self.row_indices = [item.indices()[0].int().contiguous() for item in self.coo]
        self.col_indices = [item.indices()[1].int().contiguous() for item in self.coo]
        self.values = [item.values() for item in self.coo]
        nrows = [item.size(0) for item in self.coo]
        ncols = [item.size(1) for item in self.coo]
        h_size = [self.max_h_size] * B_parts
        if B_parts * self.max_h_size != hidden_size:
            h_size[B_parts - 1] = hidden_size - (B_parts - 1) * self.max_h_size
        st = datetime.datetime.now()
        self.sp_info_ptr = torch.ops.pim_ops.spmm_coo_to_device_group(self.row_indices,
                                                                      self.col_indices,
                                                                      self.values,
                                                                      nrows,
                                                                      ncols,
                                                                      h_size,
                                                                      hidden_size)
        end = datetime.datetime.now()
        self.prepare_time = (end - st).total_seconds() * 1000
        print("[DATA]prepare_pim_time (ms): ",  self.prepare_time, flush=True)

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


def build_tune_data(hidden_size, A, sp_ds):
    row_ptr = []
    col_ind = []
    nrows = []
    ncols = []
    ds_cols = []
    ds_parts = []
    sp_ids = []
    A.col_split(sp_ds[0])
    for i, item in enumerate(A.parts):
        rowptr, col, value = item.csr()
        row_ptr.append(rowptr.int().contiguous())
        col_ind.append(col.int().contiguous())
        nrows.append(item.size(0))
        ncols.append(item.size(1))
        ds_cols.append(hidden_size // sp_ds[1])
        ds_parts.append(sp_ds[1])
        sp_ids.append(i)
    return row_ptr, col_ind, nrows, ncols, ds_cols, ds_parts, sp_ids


blnc_str = ["nnz-nnz", "nnz-row", "row-nnz", "row-row"]
sp_ds_set = [(1, 32), (2, 16)]
# sp_ds_set = [(1, 32), (2, 16), (4, 8), (8, 4), (16, 2), (32, 1)]
# def autotune(args, coo_graph: SparseTensor, split_set, blnc_set = [0, 2]):
def autotune(datadir, dataset, hidden_size, split_set, blnc_set = [0, 2]):
    # assert(args.sp_format == "CSR")
    torch.ops.load_library("./build/backend=autotuner-spf=CSR-dtype=INT32-blnc=nnz-blnc_tsklt=nnz-nr_tasklets=16-cg_lock=False-cache_size=32-sync=False/libbackend_pim.so")
    torch.ops.pim_ops.dpu_init_ranks(split_set[0][0] * split_set[0][1])
    coo_graph = load_dataset(datadir, dataset)[0]

    row_ptr = []
    col_ind = []
    nrows = []
    ncols = []
    ds_cols = []
    ds_parts = []
    sp_ids = []
    paras = []
    # print("start split the sparse matrix")
    st_split = datetime.datetime.now()
    for sp_ds in (split_set):
        # st_split = datetime.datetime.now()
        para = build_tune_data(hidden_size, SparseTensorCOO_a(coo_graph), sp_ds)
        paras.append(para)
        row_ptr += para[0]
        col_ind += para[1]
        nrows += para[2]
        ncols += para[3]
        ds_cols += para[4]
        ds_parts += para[5]
        sp_ids += para[6]
    end_split = datetime.datetime.now()
    # print("[DATA]pim_tune_split(ms): ", (end_split - st_split).total_seconds() * 1000, flush=True)

    # print(f"start tuning {len(row_ptr)} sparse partitions.")
    st_tune = datetime.datetime.now()
    tune_sp_info = torch.ops.pim_ops.prepare_tune_csr(row_ptr,
                                                      col_ind,
                                                      nrows,
                                                      ncols,
                                                      ds_cols,
                                                      ds_parts,
                                                      sp_ids,
                                                      blnc_set)
    end_tune = datetime.datetime.now()
    # print("[DATA]pim_tune_split(ms): ", (end_split - st_split).total_seconds() * 1000, flush=True)
    # print("[DATA]pim_tune_prepare(ms): ", (end_tune - st_tune).total_seconds() * 1000, flush=True)
    # print(len(row_ptr))
    # print(len(tune_sp_info))

    nr_dpus = 0
    cnt = 0
    min_time = 1e100
    min_config = [None, None, None, None, None]
    for i, sp_ds in enumerate(split_set):
        result = [0] * len(blnc_set)
        retrive_bytes = [0] * len(blnc_set)
        compute_time = [0] * len(blnc_set)

        col_size = ds_cols[i]        
        for k in range(sp_ds[0]):
            for j, blnc in enumerate(blnc_set):
                base = cnt * 8       
                # print(tune_sp_info[base:base+8])
                load_time = tune_sp_info[base+0] / HOST_DPU_BANDWIDTH
                retrive_bytes[j] += tune_sp_info[base+1]
                merge_time = tune_sp_info[base+2] / DRAM_ROW_COPY_THROUGHPUT[col_size] + (sp_ds[0] - 1) * tune_sp_info[base+2] / DRAM_ROW_ADD_THROUGHPUT[col_size]
                compute_time[j] = max(compute_time[j], tune_sp_info[base+3] / INT32_READ_FMA_THROUGHPUT[col_size])
                result[j] = load_time + merge_time
                nr_dpus = tune_sp_info[base+5]
                cnt += 1

        for j, blnc in enumerate(blnc_set):
            cloest_block_size = find_block_size(retrive_bytes[j] / nr_dpus)
            retrive_time = retrive_bytes[j] / DPU_HOST_BANDWIDTH[cloest_block_size]
            res = retrive_time + result[j] + compute_time[j]
            if res < min_time:
                min_time = res
                blnc_all = blnc_str[blnc]
                min_config = [sp_ds[0], sp_ds[1], blnc_all[:3], blnc_all[4:7], None]
            # print(f"[DATA]tune_time_total({sp_ds[0]}, {sp_ds[1]}, {blnc_str[blnc]})(ms): {result[j]:.2f})", flush=True)
    
    torch.ops.pim_ops.dpu_release()
    
    return min_config
            



    # col_size = self.hidden_size // self.dense_parts
    # col_bytes = col_size * TYPES_BYTES[args.dtype]
    # # print(col_size, HOST_DPU_BANDWIDTH, DPU_HOST_BANDWIDTH, DRAM_BANDWIDTH)
    # #[load_bytes, retrive_bytes,  merge_rows, max_nnz_per_dpu, max_nnz_per_tasklets, nr_of_dpus, kernel_read_cnt, kernel_write_cnt]
    # if self.format == "CSR":
    #     tune_sp_info = torch.ops.pim_ops.get_tune_info_csr(self.sp_info_ptr, col_bytes)
    # else:
    #     tune_sp_info = torch.ops.pim_ops.get_tune_info_coo(self.sp_info_ptr, col_bytes)
    # cloest_block_size = find_block_size(tune_sp_info[1] / tune_sp_info[5])
    # print(tune_sp_info)
    # load_time = tune_sp_info[0] / HOST_DPU_BANDWIDTH
    # retrive_time = tune_sp_info[1] / DPU_HOST_BANDWIDTH[cloest_block_size]
    # merge_time = tune_sp_info[2] / DRAM_ROW_COPY_THROUGHPUT[col_size] + (len(self.parts) - 1) * tune_sp_info[2] / DRAM_ROW_ADD_THROUGHPUT[col_size]
    # compute_time = tune_sp_info[3]  / INT32_READ_FMA_THROUGHPUT[col_size]
    # compute_time_t = tune_sp_info[4]  / (INT32_READ_FMA_THROUGHPUT[col_size] / 16)
    # write_time = tune_sp_info[7] / WRITE_LATENCY[col_bytes] / 1000
    # kernel_time = compute_time + write_time
    # kernel_time_t = compute_time_t + write_time
    # print("-------------")
    # # print("[DATA]: ", tune_sp_info[0] / 1e6,  HOST_DPU_BANDWIDTH/1e6, flush=True)
    # print("[DATA]tune_load_time(ms): ", load_time, flush=True)
    # print("[DATA]tune_retrive_time(ms): ", retrive_time, flush=True)
    # print("[DATA]tune_merge_time(ms): ", merge_time, flush=True)
    # print("[DATA]tune_kernel_time(ms): ", kernel_time, flush=True)
    # print("[DATA]tune_compute_time(ms): ", compute_time, flush=True)
    # # print("[DATA]tune_read_time(ms): ", read_time, flush=True)
    # print("[DATA]tune_write_time(ms): ", write_time, flush=True)
    # print("[DATA]tune_kernel_time_t(ms): ", kernel_time_t, flush=True)
    # print("[DATA]tune_compute_time_t(ms): ", compute_time_t, flush=True)
    # print("-------------")




def load_dataset(datadir, dataset):
    from torch_geometric.loader import RandomNodeLoader, ClusterData, ClusterLoader
    transform = T.ToSparseTensor()
    path = osp.join(datadir, dataset)
    if dataset == 'PubMed':
        dataset = Planetoid(path, "PubMed")
    elif dataset == "Reddit":
        dataset = Reddit(path)
    elif dataset == "AmazonProducts":
        dataset = datasets.AmazonProducts(path)
    elif dataset in ["ogbn-arxiv", 'ogbn-proteins', "ogbn-products"]:
        dataset = PygNodePropPredDataset(name=args.dataset, root=path)
    else:
        print("[ERROR]: load dataset failed!")
        exit(1)
    data = dataset[0]

    if data.edge_attr is None:
        data.edge_index = pyg_utils.coalesce(data.edge_index)
    else:
        data.edge_index, data.edge_attr = pyg_utils.coalesce(data.edge_index, data.edge_attr)
    if  dataset in ["AmazonProducts", "ogbn-products"]:
        import math
        num_parts = math.ceil(data.num_nodes / 500000)
        Cdata = ClusterData(data, num_parts=num_parts, save_dir=osp.join(args.datadir, args.dataset))
        data_adj_t = []
        for data in Cdata:
            data = transform(data)
            data_adj_t.append(data.adj_t)
        if args.dataset == "AmazonProducts":
            data_adj_t = [data_adj_t[1]]
    else:
        data = transform(data)
        data_adj_t = [data.adj_t]

    return data_adj_t


