import torch
import torch.nn as nn
import os
import numpy as np
import logging
import datetime
import gc
import json
import time
import torch.nn.functional as F
import warnings
import matplotlib.pyplot as plt
import random
from gcn import GCN, moco, MLP
import itertools


warnings.filterwarnings("ignore")
INF = float("inf")
_dataset = "COX2"
_nodenum = {"cora":1433, "citeseer": 3703, "pubmed": 500, "ENZYMES": 18, "COX2": 3, "BZR": 3, "PROTEINS": 1} 

train_config = {
    "gpu_id": 7,
    "dataset": _dataset,
    "epochs": 600,
    "lr": 0.0001,
    "weight_decay": 0.0001,
    "model": "GCN",  
    "save_model_dir_mask": "../ourdumps/" + _dataset + "/GCN_mask",
    "save_model_dir_context": "../ourdumps/" + _dataset + "/GCN_context",
    "seed": 0,
    "dropout": 0.5,
    "node_feature_dim": _nodenum[_dataset],
    "hidden_dim": 32,
    "gcn_hidden_dim": 32,
    "mlp_layer": 2,
}

def train(model_mask, model_context, optimizer, features, adj_normal):
    model_mask.train()
    model_context.train()

    total_time=0
    s=time.time()
    indices = adj_normal.coalesce().indices()
    values = adj_normal.coalesce().values()

    pred_mask = model_mask(features)
    pred_context = model_context(features, indices, values)

    loss = moco(pred_mask, pred_context)

    loss.requires_grad_(True)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    e=time.time()
    total_time+=e-s

    gc.collect()
    return loss.item(), total_time


def pre_train():

    torch.manual_seed(train_config["seed"])
    np.random.seed(train_config["seed"])
    torch.cuda.manual_seed(train_config["seed"]) 
    random.seed(train_config["seed"])


    save_model_dir_mask = train_config["save_model_dir_mask"]
    save_model_dir_context = train_config["save_model_dir_context"]
    os.makedirs(save_model_dir_mask, exist_ok=True)
    os.makedirs(save_model_dir_context, exist_ok=True)

    

    # set device
    device = torch.device("cuda:%d" % train_config["gpu_id"] if train_config["gpu_id"] != -1 else "cpu")
    if train_config["gpu_id"] != -1:
        torch.cuda.set_device(device)


    if train_config["model"] == "GCN":
        model_mask = MLP(train_config)
        model_context = GCN(train_config)
    if train_config["model"] == "GIN":
        model_mask = GIN(train_config)
        model_context = GIN(train_config)

    model_mask = model_mask.to(device)
    model_context = model_context.to(device)
   
    A_txt = "./dataset/" + _dataset + "/raw/" + _dataset + "_A.txt"
    src = []
    dst = []
    with open(A_txt, 'r') as file:
        for line in file:
            s, d = line.split(',')
            src.append(int(s)-1)
            dst.append(int(d)-1)
    indices = []
    indices.append(src)
    indices.append(dst)
    val_len = len(src)
    indices = torch.tensor(indices) # 74564条边

    fea_txt = "./dataset/" + _dataset + "/raw/" + _dataset + "_node_attributes.txt"
    features = []
    with open(fea_txt, 'r') as file:
        for line in file:
            float_list = [float(num) for num in line.split(',')]
            features.append(float_list)
    features = torch.tensor(features)

    graph_label_txt = "./dataset/" + _dataset + "/raw/" + _dataset + "_graph_labels.txt"
    graph_label = []
    with open(graph_label_txt, 'r') as file:
        for line in file:
            if _dataset in ["ENZYMES", "PROTEINS"]:
                graph_label.append(int(line)-1)
            elif _dataset in ["COX2", "BZR"]:
                if int(line) == -1:
                    graph_label.append(0)
                else:
                    graph_label.append(1)
    
    graph_split_txt = "./dataset/" + _dataset + "/raw/" + _dataset + "_graph_indicator.txt"
    num_counts = {}
    with open(graph_split_txt, 'r') as file:
        for line in file:
            num = int(line)
            num_counts[num] = num_counts.get(num, 0) + 1
    graph_len = [num_counts[num] for num in sorted(num_counts.keys())]

    nodenum = features.size(0)
    features = features.to(device)
    edge_index = indices

    sparse_tensor = torch.sparse_coo_tensor(
                        indices = edge_index,
                        values = torch.ones(edge_index.size(1)),
                        size=(nodenum, nodenum)
                    )
    adj_normal = sparse_tensor.coalesce()
    adj_normal = adj_normal.to(device)


    # optimizer and losses
    optimizer = torch.optim.AdamW(itertools.chain(model_mask.parameters(), model_context.parameters()), lr=train_config["lr"], weight_decay=train_config["weight_decay"],
                                  amsgrad=True)
    optimizer.zero_grad()
    best_reg_losses = INF
    best_reg_epochs = -1
    
    total_train_time=0

    plt_x=list()
    plt_y=list()

    for epoch in range(train_config["epochs"]):
        mean_reg_loss, _time = train(model_mask, model_context, optimizer, features, adj_normal)
        total_train_time+=_time
        torch.save(model_mask.state_dict(), os.path.join(save_model_dir_mask, 'epoch%d.pt' % (epoch)))
        torch.save(model_context.state_dict(), os.path.join(save_model_dir_context, 'epoch%d.pt' % (epoch)))
        if mean_reg_loss <= best_reg_losses:
            best_reg_losses = mean_reg_loss
            best_reg_epochs = epoch
            print("best mean loss: {:.3f} (epoch: {:0>3d})".format(mean_reg_loss, epoch))
        plt_x.append(epoch)
        plt_y.append(mean_reg_loss)

        plt.figure(1)
        plt.plot(plt_x,plt_y)
        plt.savefig('epoch_loss.png')
    
    best_epoch = best_reg_epochs
    model_mask.load_state_dict(torch.load(os.path.join(save_model_dir_mask, 'epoch%d.pt' % (best_epoch))))
    model_context.load_state_dict(torch.load(os.path.join(save_model_dir_context, 'epoch%d.pt' % (best_epoch))))
    torch.save(model_mask.state_dict(), os.path.join(save_model_dir_mask, "best.pt"))
    torch.save(model_context.state_dict(), os.path.join(save_model_dir_context, "best.pt"))


if __name__ == "__main__":
    pre_train()
   