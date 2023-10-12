import torch
import torch.nn as nn
import torch.nn.functional as F
import random



def pre_train(pretrain_mask, pretrain_context, X, adj, config, label_num, features):
    indices = adj.coalesce().indices()
    values = adj.coalesce().values()

    pretrain_mask.eval()
    pretrain_context.eval()

    pred_mask = pretrain_mask(features)
    pred_context = pretrain_context(X, indices, values)

    pro_pred_mask = pred_mask[-label_num:,]
    pro_pred_context = pred_context[-label_num:,]

    return pred_mask, pred_context, pro_pred_mask, pro_pred_context


class graph_prompt_structure(nn.Module):
    def __init__(self, feature, adj, prototype, num_nodes, label_num, device, trainset):
        super(graph_prompt_structure, self).__init__()
        self.weight = None
        self.label_num = label_num
        self.device = device
        self.feature = feature
        self.adj = adj
        self.prototype = prototype
        self.num_nodes = num_nodes
        self.trainset = trainset
        self.values = adj.coalesce().values()
        self.indices = adj.coalesce().indices()
        self.new_indices = self.indices.tolist()
        
    def reset_parameters(self, pre_train_model, prototype_ind, rate):
        pred = pre_train_model(self.feature.to(self.device), self.indices, self.values)
        prototype = []
        for i in range(self.label_num):
            _p = torch.mean(pred[prototype_ind[i], ], dim=0)
            prototype.append(_p.tolist())
        prototype = torch.FloatTensor(prototype).to(self.device)
        _weight = torch.mm(pred, prototype.T)

        random_indices = random.sample(range(_weight.size(0)), int(_weight.size(0)*rate))
        combined_set = set(random_indices).union(self.trainset.tolist())
        unique_list = list(combined_set)
        self.weight = torch.nn.Parameter(_weight[unique_list])

        
        repeated_list = [i for _ in range(self.label_num) for i in unique_list]
        label_list = []
        for i in range(self.label_num):
            temp = [i+self.num_nodes for j in unique_list]
            label_list += temp

        self.new_indices[0] = self.new_indices[0] + repeated_list
        self.new_indices[1] = self.new_indices[1] + label_list
        self.new_indices = torch.tensor(self.new_indices).to(self.device)

        
        X_feature = torch.zeros((self.num_nodes+self.label_num, self.feature.shape[1]))
        X_feature[:self.num_nodes, ] = self.feature
        X_feature[self.num_nodes:, ] = self.prototype

        X_feature = X_feature.to(self.device)
        
        return X_feature

    def forward(self, pre_train_mask, pre_train_context, X, config):

        new_values = self.values
        for i in range(self.label_num):
            new_values = torch.cat((new_values, F.softmax(self.weight)[:,i]), dim=0)


        sparse_tensor = torch.sparse_coo_tensor(
                            indices=self.new_indices,
                            values=new_values,
                            size=(self.num_nodes+self.label_num, self.num_nodes+self.label_num)
                        )
        sparse_tensor = sparse_tensor.coalesce()

        pred_mask, pred_context, pro_pred_mask, pro_pred_context = pre_train(pre_train_mask, pre_train_context, X, sparse_tensor, config, self.label_num, self.feature.to(self.device))
        return pred_mask, pred_context, pro_pred_mask, pro_pred_context, self.weight
