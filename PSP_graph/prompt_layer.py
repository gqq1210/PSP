import torch
import torch.nn as nn
import torch.nn.functional as F

def pre_train(pretrain_mask, pretrain_context, X, adj, config, label_num, features):
    indices = adj.coalesce().indices()
    values = adj.coalesce().values()
    adj_dense = adj.to_dense()

    if config["update_pretrain"]:
        pretrain_mask.train()
        pretrain_context.train()
    else:
        pretrain_mask.eval()
        pretrain_context.eval()

    pred_mask = pretrain_mask(features)
    pred_context = pretrain_context(X, indices, values)

    pro_pred_mask = pred_mask[-label_num:,]
    pro_pred_context = pred_context[-label_num:,]

    return pred_mask, pred_context, pro_pred_mask, pro_pred_context


class graph_prompt_structure(nn.Module):
    def __init__(self, feature, adj, prototype, num_nodes, graphnum, label_num, device, trainset, graph_len):
        super(graph_prompt_structure, self).__init__()

        self.weight= torch.nn.Parameter(torch.Tensor(graphnum, label_num))
        # self.weight = None
        self.label_num = label_num
        self.device = device
        self.feature = feature
        self.adj = adj
        self.prototype = prototype
        self.num_nodes = num_nodes
        self.graphnum = graphnum
        self.trainset = trainset
        self.graph_len = graph_len
        self.values = adj.coalesce().values()
        self.indices = adj.coalesce().indices()
        self.new_indices = self.indices.tolist()
        
    def reset_parameters(self, pre_train_model, prototype_ind):
        pred = pre_train_model(self.feature.to(self.device), self.indices, self.values)
        graph_representations = []
        start_idx = 0

        for length in self.graph_len:
            end_idx = start_idx + length
            graph_features = pred[start_idx:end_idx, :] 
            graph_representation = torch.mean(graph_features, axis=0)
            graph_representations.append(graph_representation)
            start_idx = end_idx
        graph_representations = torch.stack(graph_representations)
        graph_representations = F.normalize(graph_representations, dim=1)
        
        prototype = []
        for i in range(self.label_num):
            _p = torch.mean(graph_representations[prototype_ind[i], ], dim=0)
            prototype.append(_p.tolist())
        prototype = torch.FloatTensor(prototype).to(self.device)
        _weight = torch.mm(graph_representations, prototype.T)
        self.weight = torch.nn.Parameter(_weight)     

  

        repeated_list = []
        label_list = []
        start_idx = 0
        for length in self.graph_len:
            end_idx = start_idx + length
            extend_list = [i for i in range(start_idx, end_idx)] 
            repeated_list.extend(extend_list * self.label_num)
            start_idx = end_idx
            for j in range(self.label_num):
                label_list.extend([j] * length)
      
      
        self.new_indices[0] = self.new_indices[0] + repeated_list + label_list
        self.new_indices[1] = self.new_indices[1] + label_list + repeated_list
        self.new_indices = torch.tensor(self.new_indices).to(self.device)
        
        X_feature = torch.zeros((self.num_nodes+self.label_num, self.feature.shape[1]))
        X_feature[:self.num_nodes, ] = self.feature
        X_feature[self.num_nodes:, ] = self.prototype

        X_feature = X_feature.to(self.device)
        
        return X_feature

    def forward(self, pre_train_mask, pre_train_context, X, config):
       
        new_values = self.values
        expanded_weights = []
        for i in range(len(self.graph_len)):
            graph_weight = F.softmax(self.weight)[i] 
            expanded_weight = torch.repeat_interleave(graph_weight, self.graph_len[i], dim=0)
            expanded_weights.append(expanded_weight)

        expanded_weights = torch.cat(expanded_weights, dim=0)

        new_values = torch.cat((new_values, expanded_weights), dim=0)
        new_values = torch.cat((new_values, expanded_weights), dim=0)

        sparse_tensor = torch.sparse_coo_tensor(
                            indices=self.new_indices,
                            values=new_values,
                            size=(self.num_nodes+self.label_num, self.num_nodes+self.label_num)
                        )
        sparse_tensor = sparse_tensor.coalesce()

        pred_mask, pred_context, pro_pred_mask, pro_pred_context = pre_train(pre_train_mask, pre_train_context, X, sparse_tensor, config, self.label_num, self.feature.to(self.device))
        return pred_mask, pred_context, pro_pred_mask, pro_pred_context, self.weight


