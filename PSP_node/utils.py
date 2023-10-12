import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np
import re
import os
import json
from torch.optim.lr_scheduler import LambdaLR
from collections import OrderedDict
from multiprocessing import Pool
from tqdm import tqdm
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
import random
from sklearn.metrics import precision_recall_fscore_support

def mask_select_emb(emb,mask,device):
    index=[]
    count=0
    for i in mask:
        if i==1:
            index.append(count)
        count+=1
    index=torch.tensor(index,device=device)
    ret=torch.index_select(emb,0,index)
    return ret

##########################################################
################## Evaluation Functions ##################
##########################################################
def compute_mae(predict, count):
    error = np.absolute(predict-count)
    return error.mean()

def compute_abmae(predict, count):
    error = np.absolute(predict-count)/(count+10)
    return error.mean()

def correctness_GPU(pre, counts):
    temp=pre-counts
    # zero_positions = torch.nonzero(temp == 0).squeeze(dim=1).tolist()
    # zero_positions = [item[0] for item in zero_positions]
    # print(zero_positions)
    nonzero_num=torch.count_nonzero(temp)
    return (len(temp)-nonzero_num)/len(temp)

def correctness(pred,counts):
    return accuracy_score(counts,pred)


def microf1(pred,counts):
    return f1_score(counts,pred,average='micro')

def macrof1(pred,counts):
    return f1_score(counts,pred,average='macro')

def weightf1(pred,counts):
    return f1_score(counts,pred,average='weighted')


def compute_rmse(predict, count):
    error = np.power(predict-count, 2)
    return np.power(error.mean(), 0.5)

def compute_p_r_f1(predict, count):
    p, r, f1, _ = precision_recall_fscore_support(predict, count, average="binary")
    return p, r, f1

def compute_tp(predict, count):
    true_count = count == 1
    true_pred = predict == 1
    true_pred_count = true_count * true_pred
    return np.count_nonzero(true_pred_count) / np.count_nonzero(true_count)

def bp_compute_abmae(predict, count):
    error = torch.absolute(predict-count)/(count+1)
    return error.mean()

def max1(x):
    one=np.ones_like(x)
    return np.maximum(x,one)

def q_error(predict,count):
    predict=max1(predict)
    count=max1(count)
    return max((predict/count).mean(),(count/predict).mean())
