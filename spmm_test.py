import argparse
import datetime
import os.path as osp
import logging
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Reddit, AmazonProducts
from torch_geometric.utils import scatter

from torch_sparse import SparseTensor, matmul
from backend_pim.spmm import prepare_pim_spmm
from backend_pim.grande import prepare_pim_spmm_grande
from backend_pim.spmv import prepare_pim_spmv

device = 'cpu'


def spmm_test(data_adj_t, pim_adj_t, data_x, args):
    print("{} Dataset Info: Node({}), Edge({})".format(args.dataset, data_adj_t.size(1), data_adj_t.nnz()))
    # scale, data_x_q = symmetric_quantize(data_x)
    data_x_pim = data_x.type(args.data_type)

    start_torch = datetime.datetime.now()
    res_torch = matmul(data_adj_t, data_x)
    end_torch = datetime.datetime.now()
    print("[DATA]torch_time(ms): ", (end_torch - start_torch).total_seconds() * 1000, flush=True)

    if (args.version != "cpu"):
        start_pim = datetime.datetime.now()
        res_lib = pim_adj_t.mul(data_x_pim)
        end_pim = datetime.datetime.now()
        res_lib = res_lib.float()

        print("[DATA]pim_time_spmm(ms): ", (end_pim - start_pim).total_seconds() * 1000, flush=True)
    #     print("pim_sum: ", res_lib.sum())
    # print("torch_sum: ", res_torch.sum())


def load_datasets(args):
    path = osp.join(args.datadir, args.dataset)
    if args.dataset == 'PubMed':
        dataset = Planetoid(path, "PubMed")
    elif args.dataset == "Reddit":
        dataset = Reddit(path)
    elif args.dataset == "AmazonProducts":
        dataset = AmazonProducts(path)
    elif args.dataset in ["ogbn-arxiv", 'ogbn-proteins']:
        from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
        dataset = PygNodePropPredDataset(name=args.dataset, root=path)
    else:
        print("[ERROR]: load dataset failed!")
        exit(1)

    data = dataset[0]
    transform = T.ToSparseTensor(remove_edge_index=False)
    if args.dataset in ["AmazonProducts"]:
        from torch_geometric.loader import ClusterData
        import math
        num_parts = math.ceil(data.num_nodes / 500000)
        Cdata = ClusterData(data, num_parts=num_parts, save_dir=osp.join(args.datadir, args.dataset))
        data_parts = []
        for data in Cdata:
            data_parts.append(transform(data))
        data = data_parts[1]
    else:
        data = transform(data)


    data.x = torch.randint(-2^6, 2^6,  (data.adj_t.size(1), args.hidden_size), dtype=args.data_type)
    return data




def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', type=str, default='AmazonProducts')
    # parser.add_argument('--dataset', type=str, default='Reddit')
    parser.add_argument('--dataset', type=str, default='PubMed')
    # parser.add_argument('--dataset', type=str, default='ogbn-proteins')
    # parser.add_argument('--dataset', type=str, default='pkustk08.mtx')
    parser.add_argument('--datadir', type=str, default='./data')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--version', type=str, default='spmm', choices=["spmm", 'grande', "spmv", "cpu"])
    parser.add_argument('--tune', type=bool, default=True)
    parser.add_argument('--lib_path', type=str, default="./backend_pim/spmm_default/build/libbackend_pim.so")

    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--data_type', type=str, default='INT32', choices=["INT8", "INT32", "INT16", "INT64", "FLT32", "DBL64"])
    parser.add_argument('--sp_format', type=str, default='COO', choices=["CSR", "COO"])
    parser.add_argument('--sp_parts', type=int, default=32)
    parser.add_argument('--ds_parts', type=int, default=1)
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--nr_dpus', type=int, default=0)
    args = parser.parse_args()
    print(args, flush=True)
    TORCH_TYPES = {"INT64": torch.int64, "INT32": torch.int32, "INT16": torch.int16, "INT8": torch.int8,
                   "FLT32": torch.float32, "DBL64": torch.float64}
    args.data_type = TORCH_TYPES[args.data_type]
    return args


def main(args):
    data = load_datasets(args)

    data = data.to(device)
    pim_adj_t = None

    if (args.version != "cpu"):
        torch.ops.load_library(args.lib_path)
        if args.nr_dpus == 0:
            if args.version == "grande":
                dpus_per_rank = torch.ops.pim_ops.dpu_init_ranks(args.sp_parts)
            else:
                torch.ops.pim_ops.dpu_init_ranks(args.sp_parts * args.ds_parts)
        else:
            torch.ops.pim_ops.dpu_init_dpus(args.nr_dpus)
        if args.version == "spmm":
            pim_adj_t = prepare_pim_spmm(data.adj_t, args)
        elif args.version == "spmv":
            pim_adj_t = prepare_pim_spmv(data.adj_t, args)
        elif args.version == "grande":
            pim_adj_t = prepare_pim_spmm_grande(data.adj_t, args, dpus_per_rank)
        else:
            raise NotImplementedError



    for i in range(args.repeat):
        print("-------------------- Model=spmm_test Repeat={}--------------------".format(i), flush=True)
        spmm_test(data.adj_t, pim_adj_t, data.x, args)


    if (args.version != "cpu"):
        torch.ops.pim_ops.dpu_release()

if __name__ == "__main__":
    args = get_args()
    main(args)