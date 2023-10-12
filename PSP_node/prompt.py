import torch
import os
import numpy as np
import gc
import warnings
import torch
import torch.nn.functional as F
import numpy as np
import random
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from utils import correctness_GPU, correctness, mask_select_emb
from gcn import GCN, MLP
from prompt_layer import graph_prompt_structure
import scipy

warnings.filterwarnings("ignore")
INF = float("inf")
_dataset = "citeseer"
_nodenum = {"cora":1433, "citeseer": 3703, "pubmed": 500, "ogb": 128, "chameleon": 2325, "film": 932, "squirrel": 2089} 

train_config = {
    "gpu_id": 7,
    "epochs": 100,
    "dropout": 0.2,
    "dataset": _dataset,
    "save_pretrain_model_dir_mask": "../ourdumps/" + _dataset + "/GCN_mask",
    "save_pretrain_model_dir_context": "../ourdumps/" + _dataset + "/GCN_context",
    "downstream_save_model_dir": "../ourdumps/" + _dataset + "/Prompt/",
    "lr": 0.1,
    "weight_decay": 0.0001,
    "pretrain_model": "GCN",
    "gcn_hidden_dim": 128,
    "hidden_dim": 128,
    "mlp_layer": 2,
    "seed": 1,
    "node_feature_dim": _nodenum[_dataset],
    "rate": 1,
    "shots": 3,
}


def train(X_feature, pre_train_model_mask, pre_train_model_context, model, optimizer, device, config, label_num, trainmask, node_label):
   
    model.train()
    embedding_mask, embedding_context, pro_pred_mask, pro_pred_context, weight = model.forward(pre_train_model_mask, pre_train_model_context, X_feature, config)


    embedding_mask = embedding_mask.to(device)
    embedding_mask = mask_select_emb(embedding_mask, trainmask, device)
    embedding_context = embedding_context.to(device)
    embedding_context = mask_select_emb(embedding_context, trainmask, device)
   
    label_num=torch.tensor(label_num).to(device)

  
    sim = torch.mm(embedding_mask, pro_pred_context.t())
    pred = F.softmax(sim, dim=1)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    reg_loss = criterion(pred, node_label.squeeze().type(torch.LongTensor).to(device))
  

    reg_loss.requires_grad_(True)
    _pred = torch.argmax(pred, dim=1, keepdim=True)
    accuracy = correctness_GPU(_pred, node_label)

      
    reg_loss.backward(retain_graph=True)
    optimizer.step()
    optimizer.zero_grad()
    gc.collect()
    # print(accuracy.item())
    print("train loss", reg_loss.item())
    return reg_loss, accuracy




def evaluate(X_feature, pre_train_model_mask, pre_train_model_context, model, device, config, label_num, mask, node_label):
  
    model.eval()
    label_num=torch.tensor(label_num).to(device)
   

    embedding_mask, embedding_context, pro_pred_mask, pro_pred_context, weight = model.forward(pre_train_model_mask, pre_train_model_context, X_feature, config)
    embedding_mask = embedding_mask.to(device)
    embedding_mask = mask_select_emb(embedding_mask, mask, device)
    embedding_context = embedding_context.to(device)
    embedding_context = mask_select_emb(embedding_context, mask, device)


    sim = torch.mm(embedding_mask, pro_pred_context.t())
    pred = F.softmax(sim, dim=1)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    reg_loss = criterion(pred, node_label.squeeze().type(torch.LongTensor).to(device))


    _pred = torch.argmax(pred, dim=1, keepdim=True)
    accuracy = correctness_GPU(_pred, node_label)
    eval_pred=_pred.cpu().numpy()
    eval_graph_label=node_label.cpu().numpy()
    acc=correctness(eval_pred,eval_graph_label)
   
    gc.collect()
    # print(acc)
    return reg_loss, acc


def prompt():
    torch.manual_seed(train_config["seed"])
    np.random.seed(train_config["seed"])
    torch.cuda.manual_seed(train_config["seed"]) 
    random.seed(train_config["seed"])

    save_model_dir = train_config["downstream_save_model_dir"]
    save_pretrain_model_dir_mask = train_config["save_pretrain_model_dir_mask"]
    save_pretrain_model_dir_context = train_config["save_pretrain_model_dir_context"]
    os.makedirs(save_model_dir, exist_ok=True)

    # set device
    device = torch.device("cuda:%d" % train_config["gpu_id"] if train_config["gpu_id"] != -1 else "cpu")
    if train_config["gpu_id"] != -1:
        torch.cuda.set_device(device)


    if _dataset in ["cora", "citeseer", "pubmed"]:
        dataset = Planetoid(root='', name=_dataset, transform=NormalizeFeatures())
        data = dataset[0]
        nodenum = data.num_nodes
        trainmask = data.train_mask
        valmask = data.val_mask
        testmask = data.test_mask
        features = data.x
        nodelabel = data.y
        edge_index = data.edge_index

    elif _dataset in ['chameleon', 'film', 'squirrel']:
        fulldata = scipy.io.loadmat('./heter_data/' + _dataset + '.mat')
        edge_index = fulldata['edge_index']
        features = fulldata['node_feat']
        nodelabel = np.array(fulldata['label'], dtype=np.int).flatten()
        nodenum = features.shape[0]
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        features = torch.tensor(features, dtype=torch.float)
        nodelabel = torch.tensor(nodelabel, dtype=torch.long)

    
    nodelabelnum=nodelabel.max()+1

    if _dataset in ["cora", "citeseer", "pubmed"]:
        trainset = np.where(trainmask)[0]
        valset = np.where(valmask)[0]
        valset=torch.tensor(valset,dtype=int)
        
        newtrainset = []
        random.shuffle(trainset)
        count = train_config["shots"]
        for i in range(nodelabelnum):
            _temp = 0
            for j in trainset:
                if nodelabel[j] == i:
                    newtrainset.append(j)
                    _temp = _temp + 1
                    if _temp == count:
                        break
        trainset=torch.tensor(newtrainset,dtype=int)
        trainmask = torch.zeros(nodenum, dtype=torch.bool)
        trainmask[trainset] = True

    
    elif _dataset in ['chameleon', 'film', 'squirrel']:
        count = train_config["shots"]
        trainset = []
        valset = []
        trainnum = [0,0,0,0,0]
        valnum = [0,0,0,0,0]
        for i in range(len(nodelabel)):
            if trainnum[nodelabel[i]] < count:
                trainset.append(i)
                trainnum[nodelabel[i]] += 1
            elif valnum[nodelabel[i]] < count:
                valset.append(i)
                valnum[nodelabel[i]] += 1
            if trainnum.count(count) == 5 and valnum.count(count) == 5:
                break
        testset = list(set(range(0, features.size(0))) - set(trainset) - set(valset))
        trainmask = torch.zeros(nodenum, dtype=torch.bool)
        trainmask[trainset] = True
        valmask = torch.zeros(nodenum, dtype=torch.bool)
        valmask[valset] = True
        testmask = torch.zeros(nodenum, dtype=torch.bool)
        testmask[testset] = True
        trainset = torch.tensor(trainset)
    

    sparse_tensor = torch.sparse_coo_tensor(
                        indices = edge_index,
                        values = torch.ones(edge_index.size(1)),
                        size=(nodenum, nodenum)
                    )
    adj_normal = sparse_tensor.coalesce()
    adj_normal = adj_normal.to(device)

    

    prototype_ind = []
    for i in range(nodelabelnum):
        prototype_ind.append([])
    for i in trainset:
        prototype_ind[int(nodelabel[i])].append(int(i))
    
    prototype = []
    for i in range(nodelabelnum):
        _p = torch.mean(features[prototype_ind[i], ], dim=0)
        prototype.append(_p.tolist())
    prototype = torch.FloatTensor(prototype)

    pre_train_model_mask = MLP(train_config)
    pre_train_model_context = GCN(train_config)
 
    pre_train_model_mask = pre_train_model_mask.to(device)
    pre_train_model_context = pre_train_model_context.to(device)

    pre_train_model_mask.load_state_dict(torch.load(os.path.join(save_pretrain_model_dir_mask, 'best.pt')))
    pre_train_model_context.load_state_dict(torch.load(os.path.join(save_pretrain_model_dir_context, 'best.pt')))

  
    
    model = graph_prompt_structure(features, adj_normal, prototype, nodenum, nodelabelnum, device, trainset)
    model = model.to(device)
    X_feature = model.reset_parameters(pre_train_model_context, prototype_ind, train_config["rate"])
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config["lr"],
                                        weight_decay=train_config["weight_decay"], amsgrad=True)

    optimizer.zero_grad()
    best_reg_epochs = {"train": -1, "dev": -1, "test": -1}
    best_acc = {"train": -1, "dev": -1, "test": -1}
   
    nodelabel = nodelabel.to(device)

    trainmask, valmask, testmask = trainmask.to(device), valmask.to(device), testmask.to(device)
    trainlabel = torch.masked_select(nodelabel, torch.tensor(trainmask, dtype=bool)).unsqueeze(1)
    vallabel = torch.masked_select(nodelabel, torch.tensor(valmask, dtype=bool)).unsqueeze(1)
    testlabel = torch.masked_select(nodelabel, torch.tensor(testmask, dtype=bool)).unsqueeze(1)


    for epoch in range(train_config["epochs"]):
        mean_reg_loss, accfold = train(X_feature, pre_train_model_mask, pre_train_model_context, model, optimizer, 
                                        device, train_config, nodelabelnum, trainmask, trainlabel)
        torch.save(model.state_dict(), os.path.join(save_model_dir, 'epoch%d.pt' % (epoch)))
            
        if accfold >= best_acc["train"]:
            best_acc["train"] = accfold
            best_reg_epochs["train"] = epoch
            

        mean_reg_loss, accfold = evaluate(X_feature, pre_train_model_mask, pre_train_model_context, model, 
                                          device, train_config, nodelabelnum, valmask, vallabel)
        if accfold >= best_acc["dev"]:
            best_acc["dev"] = accfold
            best_reg_epochs["dev"] = epoch


    
    best_epoch = best_reg_epochs["dev"]
    model.load_state_dict(torch.load(os.path.join(save_model_dir, 'epoch%d.pt' % (best_epoch))))
    mean_reg_loss, acctest = evaluate(X_feature, pre_train_model_mask, pre_train_model_context, model, 
                                      device, train_config, nodelabelnum, testmask, testlabel)

    print("testacc:", acctest*100)
    # print("train " + str(best_reg_epochs["train"]) + " " + "val " + str(best_reg_epochs["dev"]))
    return acctest*100



if __name__ == "__main__":     
    prompt()
    