import argparse
import datetime
import os.path as osp
import logging
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Reddit, AmazonProducts
from torch_geometric.utils import scatter
from torch_geometric.logging import init_wandb, log
from models.models import GCN, SAGE, GIN
import torch_geometric.utils as pyg_utils
from sklearn.metrics import f1_score
from backend_pim.spmm import prepare_pim_spmm
from backend_pim.grande import prepare_pim_spmm_grande
from backend_pim.spmv import prepare_pim_spmv

device = 'cpu'

@torch.no_grad()
def test(args, model, data, argmax=True,evaluator=None):
    model.eval()
    st = datetime.datetime.now()
    y_pred = model(data["x"], data["adj_t"], data["edge_attr"])
    end = datetime.datetime.now()
    print("[DATA]infer_time(ms): ", (end-st).total_seconds() * 1000)
    if argmax:
        y_pred = y_pred.argmax(dim=-1, keepdim=True)

    y_ture = data["y"]
    if args.dataset.startswith("ogbn"):
        test_acc = evaluator.eval({
            'y_true': y_ture,
            'y_pred': y_pred,
        })[evaluator.eval_metric]
    else:
        y_pred = y_pred.squeeze(dim=1)
        # out = y_pred
        test_acc = (y_pred.eq(y_ture).sum() / y_pred.size(0)).item()

    return test_acc


def load_datasets(args):
    path = osp.join(args.datadir, args.dataset)
    evaluator = None
    if args.dataset == 'PubMed':
        dataset = Planetoid(path, "PubMed")
    elif args.dataset == "Reddit":
        dataset = Reddit(path)
    elif args.dataset == "AmazonProducts":
        dataset = AmazonProducts(path)
    elif args.dataset in ["ogbn-arxiv", 'ogbn-proteins']:
        from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
        dataset = PygNodePropPredDataset(name=args.dataset, root=path)
        evaluator = Evaluator(args.dataset)
    else:
        print("[ERROR]: load dataset failed!")
        exit(1)

    data = dataset[0]
    argmax = True
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
        data.y = data.y.argmax(dim=-1)
    else:
        data = transform(data)

    if args.dataset.startswith("ogbn"):
        data.node_species = None
        if args.dataset == "ogbn-proteins":
            data.y = data.y.to(torch.float)
            # Initialize features of nodes by aggregating edge features.
            row, col = data.edge_index
            data.x = scatter(data.edge_attr, col, dim_size=data.num_nodes, reduce='sum')
            argmax = False
            num_classes = data.y.size(-1)
        else:
            num_classes = dataset.num_classes
    else:
        num_classes = dataset.num_classes

    return data, evaluator, argmax, num_classes




def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', type=str, default='AmazonProducts')
    # parser.add_argument('--dataset', type=str, default='Reddit')
    parser.add_argument('--dataset', type=str, default='PubMed')
    # parser.add_argument('--dataset', type=str, default='ogbn-proteins')
    # parser.add_argument('--dataset', type=str, default='pkustk08.mtx')
    parser.add_argument('--datadir', type=str, default='../data')
    parser.add_argument('--model', type=str, default='gcn', choices=["gcn", "gin", "sage"])
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--version', type=str, default='grande', choices=["spmm", 'grande', "spmv", "cpu"])
    # parser.add_argument('--devices', type=str, default="pim", choices=["cpu", "pim"])
    # parser.add_argument('--lib_path', type=str, default='None')
    parser.add_argument('--lib_path', type=str, default="./backend_pim/spmm_grande/build/libbackend_pim.so")

    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--data_type', type=str, default='INT32', choices=["INT8", "INT32", "INT16", "INT64", "FLT32", "DBL64"])
    parser.add_argument('--sp_format', type=str, default='CSR', choices=["CSR", "COO"])
    parser.add_argument('--sp_parts', type=int, default=2)
    parser.add_argument('--ds_parts', type=int, default=16)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--nr_dpus', type=int, default=0)
    args = parser.parse_args()
    print(args, flush=True)
    TORCH_TYPES = {"INT64": torch.int64, "INT32": torch.int32, "INT16": torch.int16, "INT8": torch.int8,
                   "FLT32": torch.float32, "DBL64": torch.float64}
    args.data_type = TORCH_TYPES[args.data_type]
    return args


def main(args):
    data, evaluator, argmax, num_classes = load_datasets(args)

    data = data.to(device)
    data_dict = {"x": data.x, "y": data.y, "edge_attr": data.edge_attr}

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
            data_dict['adj_t'] = prepare_pim_spmm(data.adj_t, args)
        elif args.version == "spmv":
            data_dict['adj_t'] = prepare_pim_spmv(data.adj_t, args)
        elif args.version == "grande":
            data_dict['adj_t'] = prepare_pim_spmm_grande(data.adj_t, args, dpus_per_rank)
        else:
            raise NotImplementedError
    else:
        data_dict['adj_t'] = data.adj_t

    nr_layers = args.num_layers
    if args.model == "gcn":
        model = GCN(data.x.size(-1), args.hidden_size, num_classes, nr_layers)
    elif args.model == "gin":
        model = GIN(data.x.size(-1), args.hidden_size, num_classes, nr_layers)
    elif args.model == "sage":
        model = SAGE(data.x.size(-1), args.hidden_size, num_classes, nr_layers)
    else:
        raise NotImplementedError

    model = model.to(device)

    for i in range(args.repeat):
        print("-------------------- Model={} nrl={} Repeat={}--------------------".format(args.model, nr_layers, i), flush=True)
        tmp_test_acc = test(args, model, data_dict, argmax, evaluator)
        print(f"Test_acc: {tmp_test_acc:.4f}")

    if (args.version != "cpu"):
        torch.ops.pim_ops.dpu_release()

if __name__ == "__main__":
    args = get_args()
    main(args)