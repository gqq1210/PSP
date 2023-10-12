import torch
import os
import numpy as np
import datetime
import gc
import warnings
import matplotlib.pyplot as plt
import random
from gcn import GCN, moco, MLP
import itertools
import dgl

warnings.filterwarnings("ignore")
INF = float("inf")
_dataset = "ENZYMES"
train_config = {
    "gpu_id": 7,
    "mask_rate": 0,
    "dataset": _dataset,
    "epochs": 600,
    "pretrain_hop_num": 1,
    "lr": 0.0001,
    "weight_decay": 0.0001,
    "model": "GCN",  
    "save_model_dir_mask": "../ourdumps/" + _dataset + "/GCN_mask",
    "save_model_dir_context": "../ourdumps/" + _dataset + "/GCN_context",
    "seed": 0,
    "dropout": 0.5,
    "node_feature_dim": 18,
    "hidden_dim": 32,
    "gcn_hidden_dim": 32,
    "mlp_layer": 2,
}

def train(model_mask, model_context, optimizer, features, adj_normal):
    model_mask.train()
    model_context.train()

    indices = adj_normal.coalesce().indices()
    values = adj_normal.coalesce().values()

    pred_mask = model_mask(features)
    pred_context = model_context(features, indices, values)
    loss = moco(pred_mask, pred_context)

    loss.requires_grad_(True)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    gc.collect()
    return loss.item()


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


    model_mask = MLP(train_config)
    model_context = GCN(train_config)

    model_mask = model_mask.to(device)
    model_context = model_context.to(device)
   
    root_fewshot_dir = "./dataset/ENZYMES/1_trainshot_1_valshot_10_tasks"
    num = 0
    fewshot_dir=os.path.join(root_fewshot_dir,str(num))
    trainset = np.load(os.path.join(fewshot_dir, "train_dgl_dataset.npy"),allow_pickle=True)
    valset = np.load(os.path.join(fewshot_dir, "val_dgl_dataset.npy"),allow_pickle=True)
    testset = np.load(os.path.join(fewshot_dir, "test_dgl_dataset.npy"),allow_pickle=True)
    trainset=torch.tensor(trainset,dtype=int)[0]
    valset=torch.tensor(valset,dtype=int)[0]
    testset=torch.tensor(testset,dtype=int)[0]
    graph=dgl.load_graphs(os.path.join("./dataset/ENZYMES/all/",str(num)))[0][0]
    nodelabel=graph.ndata["label"]
    nodenum=graph.number_of_nodes()
    nodelabelnum=nodelabel.max()+1
    features = graph.ndata["feature"]
    edge_index = graph.edges()
    edge_index = torch.stack(edge_index, dim=0)

    features = features.to(device)

    sparse_tensor = torch.sparse_coo_tensor(
                        indices = edge_index,
                        values = torch.ones(edge_index.size(1)),
                        size=(nodenum, nodenum)
                    )
    adj_normal = sparse_tensor.coalesce()
    adj_normal = adj_normal.to(device)


    optimizer = torch.optim.AdamW(itertools.chain(model_mask.parameters(), model_context.parameters()), lr=train_config["lr"], weight_decay=train_config["weight_decay"],
                                  amsgrad=True)
    optimizer.zero_grad()
    best_reg_losses = INF
    best_reg_epochs = -1


    plt_x=list()
    plt_y=list()

    for epoch in range(train_config["epochs"]):
        mean_reg_loss = train(model_mask, model_context, optimizer, features, adj_normal)
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