import torch
import os
import numpy as np
import logging
import datetime
import gc
import time
import warnings
import random
import torch.nn.functional as F
from utils import correctness_GPU,correctness,macrof1,weightf1,center_embedding, mask_select_emb

from gcn import GCN, MLP
from prompt_layer import graph_prompt_structure
from collections import defaultdict

warnings.filterwarnings("ignore")
INF = float("inf")
_dataset = "COX2"
_nodenum = {"cora":1433, "citeseer": 3703, "pubmed": 500, "ENZYMES": 18, "COX2": 3, "BZR": 3, "PROTEINS": 1} 

train_config = {
    "gpu_id": 7,
    "epochs": 50,
    "dropout": 0.8,
    "dataset": _dataset,
    "save_pretrain_model_dir_mask": "../ourdumps/" + _dataset + "/GCN_mask",
    "save_pretrain_model_dir_context": "../ourdumps/" + _dataset + "/GCN_context",
    "downstream_save_model_dir": "../ourdumps/" + _dataset + "/Prompt",
    "lr": 0.1,
    "weight_decay": 0.0001,
    "pretrain_model": "GCN",
    "gcn_hidden_dim": 32,
    "hidden_dim": 32,
    "mlp_layer": 2,
    "seed": 0,
    "update_pretrain": False,
    "node_feature_dim": _nodenum[_dataset],
    "shots": 5,
}


def train(X_feature, pre_train_model_mask, pre_train_model_context, model, optimizer, device, config, label_num, trainmask, label, graph_len):
   
    model.train()
    embedding_mask, embedding_context, pro_pred_mask, pro_pred_context, weight = model.forward(pre_train_model_mask, pre_train_model_context, X_feature, config)

    graph_representations_mask = []
    start_idx = 0
    for length in graph_len:
        end_idx = start_idx + length
        graph_features = embedding_mask[start_idx:end_idx, :]  
        graph_representation = torch.mean(graph_features, axis=0)  
        graph_representations_mask.append(graph_representation)
        start_idx = end_idx
    graph_representations_mask = torch.stack(graph_representations_mask)

    graph_representations_context = []
    start_idx = 0
    for length in graph_len:
        end_idx = start_idx + length
        graph_features = embedding_context[start_idx:end_idx, :]  
        graph_representation = torch.mean(graph_features, axis=0) 
        graph_representations_context.append(graph_representation)
        start_idx = end_idx
    graph_representations_context = torch.stack(graph_representations_context)


    embedding_mask = graph_representations_mask.to(device)
    embedding_mask = mask_select_emb(embedding_mask, trainmask, device)

    embedding_context = graph_representations_context.to(device)
    embedding_context = mask_select_emb(embedding_context, trainmask, device)

    label_num=torch.tensor(label_num).to(device)
    sim = torch.mm(embedding_mask, pro_pred_context.t())
    pred = F.softmax(sim, dim=1)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    reg_loss = criterion(pred, label.squeeze().type(torch.LongTensor).to(device))
    reg_loss.requires_grad_(True)
    _pred = torch.argmax(pred, dim=1, keepdim=True)
    accuracy = correctness_GPU(_pred, label)

      
    reg_loss.backward(retain_graph=True)
    optimizer.step()
    optimizer.zero_grad()
    gc.collect()
    # print(accuracy.item())
    # print("train loss", reg_loss.item())
    return reg_loss, accuracy




def evaluate(X_feature, pre_train_model_mask, pre_train_model_context, model, device, config, label_num, mask, label, graph_len):
  
    model.eval()
    label_num=torch.tensor(label_num).to(device)
    
    embedding_mask, embedding_context, pro_pred_mask, pro_pred_context, weight = model.forward(pre_train_model_mask, pre_train_model_context, X_feature, config)
    
    graph_representations_mask = []
    start_idx = 0
    for length in graph_len:
        end_idx = start_idx + length
        graph_features = embedding_mask[start_idx:end_idx, :] 
        graph_representation = torch.mean(graph_features, axis=0)  
        graph_representations_mask.append(graph_representation)
        start_idx = end_idx
    graph_representations_mask = torch.stack(graph_representations_mask)

    graph_representations_context = []
    start_idx = 0
    for length in graph_len:
        end_idx = start_idx + length
        graph_features = embedding_context[start_idx:end_idx, :] 
        graph_representation = torch.mean(graph_features, axis=0) 
        graph_representations_context.append(graph_representation)
        start_idx = end_idx
    graph_representations_context = torch.stack(graph_representations_context)
    
    embedding_mask = graph_representations_mask.to(device)
    embedding_mask = mask_select_emb(embedding_mask, mask, device)

    embedding_context = graph_representations_context.to(device)
    embedding_context = mask_select_emb(embedding_context, mask, device)

    sim = torch.mm(embedding_mask, pro_pred_context.t())

    pred = F.softmax(sim, dim=1)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    reg_loss = criterion(pred, label.squeeze().type(torch.LongTensor).to(device))


    _pred = torch.argmax(pred, dim=1, keepdim=True)
    accuracy = correctness_GPU(_pred, label)
    eval_pred=_pred.cpu().numpy()
    eval_graph_label=label.cpu().numpy()
    acc=correctness(eval_pred,eval_graph_label)
    gc.collect()
    # print(acc)
    # print(reg_loss.item())
    return reg_loss, acc



def prompt():
    torch.manual_seed(train_config["seed"])
    np.random.seed(train_config["seed"])
    torch.cuda.manual_seed(train_config["seed"]) 
    random.seed(train_config["seed"])

    trainstart = train_config["shots"]
    save_model_dir = train_config["downstream_save_model_dir"]
    save_pretrain_model_dir_mask = train_config["save_pretrain_model_dir_mask"]
    save_pretrain_model_dir_context = train_config["save_pretrain_model_dir_context"]
    os.makedirs(save_model_dir, exist_ok=True)


    # set device
    device = torch.device("cuda:%d" % train_config["gpu_id"] if train_config["gpu_id"] != -1 else "cpu")
    if train_config["gpu_id"] != -1:
        torch.cuda.set_device(device)

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
    graph_label = torch.tensor(graph_label)
    
    graph_split_txt = "./dataset/" + _dataset + "/raw/" + _dataset + "_graph_indicator.txt"
    num_counts = {}
    with open(graph_split_txt, 'r') as file:
        for line in file:
            num = int(line)
            num_counts[num] = num_counts.get(num, 0) + 1
    graph_len = [num_counts[num] for num in sorted(num_counts.keys())]
    # graph_len = torch.tensor(graph_len).to(device)

    
    nodenum = features.size(0)
    features = features.to(device)
    edge_index = indices
    labelnum = graph_label.max()+1
    graphnum = len(graph_label)

    class_indices = defaultdict(list)
    for idx, class_label in enumerate(graph_label):
        class_indices[class_label.item()].append(idx)

    train_indices, val_indices, test_indices = [], [], []
    for class_label, indices in class_indices.items():
        train_indices.extend(indices[trainstart : trainstart + train_config["shots"]])
        val_indices.extend(indices[trainstart + train_config["shots"] : trainstart + 2*train_config["shots"]])
        test_indices.extend(indices[trainstart + 2*train_config["shots"] : ])
        test_indices.extend(indices[: trainstart])


    trainset = torch.tensor(train_indices)
    valset = torch.tensor(val_indices)
    testset = torch.tensor(test_indices)
    trainmask = [True if i in trainset else False for i in range(graphnum)]
    valmask = [True if i in valset else False for i in range(graphnum)]
    testmask = [not (train or val) for train, val in zip(trainmask, valmask)]

   
    trainset = np.where(trainmask)[0]
    trainset=torch.tensor(trainset,dtype=int)
    trainmask = torch.tensor(trainmask)
    valmask = torch.tensor(valmask)
    testmask = torch.tensor(testmask)

    sparse_tensor = torch.sparse_coo_tensor(
                        indices = edge_index,
                        values = torch.ones(edge_index.size(1)),
                        size=(nodenum, nodenum)
                    )
    adj_normal = sparse_tensor.coalesce()
    adj_normal = adj_normal.to(device)

   
    prototype_ind = []
    for i in range(labelnum):
        prototype_ind.append([])
    for i in trainset:
        prototype_ind[int(graph_label[i])].append(int(i))


    graph_representations = []
    start_idx = 0
    for length in graph_len:
        end_idx = start_idx + length
        graph_features = features[start_idx:end_idx, :] 
        graph_representation = torch.mean(graph_features, axis=0) 
        graph_representations.append(graph_representation)
        start_idx = end_idx
    graph_representations = torch.stack(graph_representations)
    graph_representations = F.normalize(graph_representations, dim=1)

    prototype = []
    for i in range(labelnum):
        _p = torch.mean(graph_representations[prototype_ind[i], ], dim=0)
        prototype.append(_p.tolist())
    prototype = torch.FloatTensor(prototype)
    prototype = F.normalize(prototype)

 

    if train_config["pretrain_model"] == "GCN":
        pre_train_model_mask = MLP(train_config)
        pre_train_model_context = GCN(train_config)
    if train_config["pretrain_model"] == "GIN":
        pre_train_model_mask = GIN(train_config)
        pre_train_model_context = GIN(train_config)

    pre_train_model_mask = pre_train_model_mask.to(device)
    pre_train_model_context = pre_train_model_context.to(device)

    pre_train_model_mask.load_state_dict(torch.load(os.path.join(save_pretrain_model_dir_mask, 'best.pt')))
    pre_train_model_context.load_state_dict(torch.load(os.path.join(save_pretrain_model_dir_context, 'best.pt')))

   
    
    model = graph_prompt_structure(features, adj_normal, prototype, nodenum, graphnum, labelnum, device, trainset, graph_len)
    model = model.to(device)
    X_feature = model.reset_parameters(pre_train_model_context, prototype_ind)
   
    
    # optimizer and losses
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config["lr"], weight_decay=train_config["weight_decay"], amsgrad=True)

    optimizer.zero_grad()
    best_reg_epochs = {"train": -1, "dev": -1, "test": -1}
    best_acc = {"train": -1, "dev": -1, "test": -1}
   
    graph_label = graph_label.to(device)

    trainmask, valmask, testmask = trainmask.to(device), valmask.to(device), testmask.to(device)
    trainlabel = torch.masked_select(graph_label, torch.tensor(trainmask, dtype=bool)).unsqueeze(1)
    vallabel = torch.masked_select(graph_label, torch.tensor(valmask, dtype=bool)).unsqueeze(1)
    testlabel = torch.masked_select(graph_label, torch.tensor(testmask, dtype=bool)).unsqueeze(1)

    # 在epoch里面train  pretrain model GIN
    for epoch in range(train_config["epochs"]):
        mean_reg_loss, accfold = train(X_feature, pre_train_model_mask, pre_train_model_context, model, optimizer,
                                       device, train_config, labelnum, trainmask, trainlabel, graph_len)
        torch.save(model.state_dict(), os.path.join(save_model_dir, 'epoch%d.pt' % (epoch)))
        if accfold >= best_acc["train"]:
            best_acc["train"] = accfold
            best_reg_epochs["train"] = epoch
            

        mean_reg_loss, accfold = evaluate(X_feature, pre_train_model_mask, pre_train_model_context, model, 
                                          device, train_config, labelnum, valmask, vallabel, graph_len)
        if accfold >= best_acc["dev"]:
            best_acc["dev"] = accfold
            best_reg_epochs["dev"] = epoch
        _, acctest = evaluate(X_feature, pre_train_model_mask, pre_train_model_context, model, 
                              device, train_config,labelnum, testmask, testlabel, graph_len)

    best_epoch = best_reg_epochs["dev"]
    model.load_state_dict(torch.load(os.path.join(save_model_dir, 'epoch%d.pt' % (best_epoch))))
    mean_reg_loss, acctest = evaluate(X_feature, pre_train_model_mask, pre_train_model_context, model,
                                    device, train_config, labelnum, testmask, testlabel, graph_len)

    print("testacc:", acctest*100)
    print("train " + str(best_reg_epochs["train"]) + " " + "val " + str(best_reg_epochs["dev"]))
    return acctest*100

if __name__ == "__main__": 
    prompt()
    