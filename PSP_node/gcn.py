import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.in_channels = config["node_feature_dim"]
        self.hidden_channels = config["hidden_dim"]
        self.out_channels = config["gcn_hidden_dim"]
        self.num_layers = config["mlp_layer"]
        self.dropout = config["dropout"]
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        if self.num_layers == 1:
            # just linear layer i.e. logistic regression
            self.lins.append(nn.Linear(self.in_channels, self.out_channels))
        else:
            self.lins.append(nn.Linear(self.in_channels, self.hidden_channels))
            self.bns.append(nn.BatchNorm1d(self.hidden_channels))
            for _ in range(self.num_layers - 2):
                self.lins.append(nn.Linear(self.hidden_channels, self.hidden_channels))
                self.bns.append(nn.BatchNorm1d(self.hidden_channels))
            self.lins.append(nn.Linear(self.hidden_channels, self.out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x
    

class GCN(nn.Module):
    def __init__(self, config):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(config["node_feature_dim"], config["gcn_hidden_dim"])
        self.conv2 = GCNConv(config["gcn_hidden_dim"], config["gcn_hidden_dim"])
        self.dropout = config["dropout"]
        
    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.normalize(x, dim=1)
        return x



def moco(z1, z2, tau=0.5):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    dots = torch.exp(torch.mm(z1, z2.t()) / tau)
    nominator = torch.diag(dots)
    denominator1 = (dots.sum(axis=0) ) 
    denominator2 = (dots.sum(axis=1) )
    loss_con = ((-1) * (torch.log(nominator / denominator1))).mean() + ((-1) * (torch.log(nominator / denominator2))).mean()
    loss_con = loss_con / 2.0
    return loss_con
