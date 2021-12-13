import os.path as osp
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import random
import math
import numpy as np
from config.CurvGN_config import Config
from CurvGN_module import ConvCurv
from data_preprocess.prepare_GCN_data import *
#load the neural networks
from torch_scatter import scatter

def loss_function(q, k, v, τ=0.5):
    # N是batch size
    N = q.shape[0]
    # C是 vector dim
    C = q.shape[1]
    pos = torch.exp(torch.div(torch.bmm(q.view(N, 1, C), k.view(N, C, 1)).view(N, 1), τ))  # NX1
    neg = v
    # 求和
    denominator = neg + pos
    return torch.mean(-torch.log(torch.div(pos,denominator)))  # scalar
def train(train_mask):
    model.train()
    optimizer.zero_grad()
    if config.method == "ConvCurv":
        nll_loss, Reg1, Reg2 = model(data)
        cross_entropy_loss = F.nll_loss(nll_loss[train_mask], data.y[train_mask])
        loss = cross_entropy_loss + config.gamma1 * Reg1 + config.gamma2 * Reg2 if config.loss_mode == 1 else cross_entropy_loss
    else:
        nll_loss, embeddings = model(data)
        cross_entropy_loss = F.nll_loss(nll_loss[train_mask], data.y[train_mask])
        q = torch.index_select(embeddings, 0, data.pos_edge_index0)
        k = torch.index_select(embeddings, 0, data.pos_edge_index1)
        v_0 = torch.index_select(embeddings, 0, data.neg_edge_index0)
        v_1 = torch.index_select(embeddings, 0, data.neg_edge_index1)
        v = torch.exp(torch.div(torch.bmm(v_0.view(v_0.shape[0], 1, v_0.shape[1]), v_1.view(v_1.shape[0], v_1.shape[1], 1)).view(v_1.shape[0], 1), τ))
        v = scatter(v, data.neg_edge_index0, dim=0)
        v = torch.index_select(v, 0, data.pos_edge_index0)
        loss = loss_function(q, k, v) + cross_entropy_loss
    loss.backward()
    optimizer.step()

def test(train_mask,val_mask,test_mask):
    model.eval()
    if config.method == "ConvCurv":
        logits, Reg1, Reg2 = model(data)
    else:
        logits, _ = model(data)
    accs = []
    for mask in [train_mask, val_mask, test_mask]:
        pred = logits[mask].max(1)[1]
        #print(pred)
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    accs.append(F.nll_loss(logits[val_mask], data.y[val_mask]))
    # print(accs)
    return accs

config = Config()
print(config.__dict__)
#load dataset
times = range(config.times)  #Todo:实验次数
epoch_num = config.epoch_num
wait_total = config.patience
pipelines = [config.method]
# d_names=['Cora','Citeseer','PubMed']
d_names = config.d_names

train_dataset, test_dataset = get_GCN_data("CUB")
for time in times:
    train_mask = train_dataset.train_mask.bool()
    # val_mask=data.val_mask.bool()
    test_mask = test_dataset.test_mask.bool()
    model, data = ConvCurv.call(train_dataset, config.d_names, train_dataset.x.size(1), train_dataset.num_classes, config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=0.0005)
    best_val_acc = test_acc = 0.0
    best_val_loss = np.inf
    for epoch in range(0, epoch_num):
        train(train_mask)
        train_acc, val_acc, tmp_test_acc, val_loss = test(train_mask, val_mask, test_mask)
        if val_acc >= best_val_acc:
            test_acc = tmp_test_acc
            best_val_acc = val_acc
            best_val_loss = val_loss
            wait_step = 0
        else:
            wait_step += 1
            if wait_step == wait_total:
                print('Early stop! Min loss: ', best_val_loss, ', Max accuracy: ', best_val_acc)
                break