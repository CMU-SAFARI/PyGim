import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d as BatchNorm
from torch.nn import Linear, ReLU, Sequential
from pyg_gcn_conv import GCNConv
from pyg_sage_conv import SAGEConv
from pyg_gin_conv import GINConv
from torch_geometric.nn import MLP
from tqdm import tqdm


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.ln1 = Linear(in_channels, hidden_channels)
        self.bn0 = torch.nn.BatchNorm1d(hidden_channels)

        self.bns = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=False))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        self.ln2 = Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def reset_parameters(self):
        self.ln1.reset_parameters()
        self.ln2.reset_parameters()
        self.bn0.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t, edge_attr):
        x = self.ln1(x)
        x = self.bn0(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        for i, conv in enumerate(self.convs):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.ln2(x)
        return x

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.ln1 = Linear(in_channels, hidden_channels)
        self.bn0 = torch.nn.BatchNorm1d(hidden_channels)

        self.bns = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                SAGEConv(hidden_channels, hidden_channels, cached=False))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        self.ln2 = Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def reset_parameters(self):
        self.ln1.reset_parameters()
        self.ln2.reset_parameters()
        self.bn0.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t, edge_attr = None):
        x = self.ln1(x)
        x = self.bn0(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        for i, conv in enumerate(self.convs):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.ln2(x)
        return x


class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5):
        super(GIN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.ln1 = Linear(in_channels, hidden_channels)
        self.bn0 = torch.nn.BatchNorm1d(hidden_channels)

        self.bns = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                GINConv(MLP([hidden_channels, hidden_channels, hidden_channels])))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        self.ln2 = Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def reset_parameters(self):
        self.ln1.reset_parameters()
        self.ln2.reset_parameters()
        self.bn0.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t, edge_attr = None):
        x = self.ln1(x)
        x = self.bn0(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        for i, conv in enumerate(self.convs):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.ln2(x)
        return x

