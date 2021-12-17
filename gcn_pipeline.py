import os.path as osp
import os
os.environ['CUDA_VISIBLE_DEVICES']='2,3'
import numpy
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
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from imblearn.under_sampling import RandomUnderSampler

τ = 0.5
def get_qkv_dataset(embeddings, pos_edge_index_0, pos_edge_index_1, neg_edge_index_0, neg_edge_index_1, mode):
    embeddings = F.normalize(embeddings, dim=-1)
    q = torch.index_select(embeddings, 0, pos_edge_index_0)
    k = torch.index_select(embeddings, 0, pos_edge_index_1)
    v_0 = torch.index_select(embeddings, 0, neg_edge_index_0)
    v_1 = torch.index_select(embeddings, 0, neg_edge_index_1)
    v = torch.exp(torch.div(torch.bmm(v_0.view(v_0.shape[0], 1, v_0.shape[1]), v_1.view(v_1.shape[0], v_1.shape[1], 1)).view(v_1.shape[0], 1), τ))
    v = scatter(v, neg_edge_index_0, dim=0)
    v = torch.index_select(v, 0, pos_edge_index_0)
    pos_labels = torch.ones(q.size(0), dtype=torch.long)
    neg_labels = torch.zeros(v_0.size(0), dtype=torch.long)
    labels = torch.cat((pos_labels, neg_labels), dim=0)
    rus = RandomUnderSampler(random_state=0)
    index, labels = rus.fit_resample(torch.arange(labels.size(0)).view(-1, 1), labels)
    index = index.reshape(-1)
    pos_pairs = torch.cat((q, k), dim=1)
    neg_pairs = torch.cat((v_0, v_1), dim=1)
    pairs = torch.cat((pos_pairs, neg_pairs), dim=0)
    pairs = pairs[index]

    sim_pairs = None
    if mode == 1:
        index_0 = torch.cat((pos_edge_index_0, neg_edge_index_0), dim=0)
        index_1 = torch.cat((pos_edge_index_1, neg_edge_index_1), dim=0)
        index_0 = index_0[index]
        index_1 = index_1[index]
        sim_index_0 = torch.index_select(test_dataset.x, 0, index_0)
        sim_index_1 = torch.index_select(test_dataset.x, 0, index_1)
        sim_pairs = torch.cat((sim_index_0, sim_index_1), dim=1)
    return q, k, v, pairs, torch.tensor(labels).cuda(), sim_pairs


def loss_function(q, k, v, gamma=1e-10):
    # N是batch size
    N = q.shape[0]
    # C是 vector dim
    C = q.shape[1]
    pos = torch.exp(torch.div(torch.bmm(q.view(N, 1, C), k.view(N, C, 1)).view(N, 1), τ))  # NX1
    neg = v
    # 求和
    denominator = neg + pos
    item_loss = -torch.log(torch.div(pos, denominator) + gamma)
    infoNCE_loss = torch.mean(item_loss)
    if torch.isnan(infoNCE_loss):
        raise ValueError
    return infoNCE_loss  # scalar
def train(train_mask):
    model.train()
    optimizer.zero_grad()
    if config.method != "ConvCurv":
        nll_loss = model(train_dataset)
        cross_entropy_loss = F.nll_loss(nll_loss[train_mask], data.y[train_mask])
        loss = cross_entropy_loss + config.gamma1 * Reg1 + config.gamma2 * Reg2 if config.loss_mode == 1 else cross_entropy_loss
    else:
        _, embeddings = model(train_dataset)
        # cross_entropy_loss = F.nll_loss(nll_loss[train_mask], data.y[train_mask])
        q, k, v, pairs, labels, _ = get_qkv_dataset(embeddings, train_dataset.pos_edge_index_0, train_dataset.pos_edge_index_1, train_dataset.neg_edge_index_0, train_dataset.neg_edge_index_1, 0)
        logits = model.fc(pairs)
        nll_loss = F.log_softmax(logits, dim=1)
        cross_entropy_loss = F.nll_loss(nll_loss, labels)
        _, predictions = torch.max(nll_loss.data, 1)
        info_nce_loss = loss_function(q, k, v)
        loss = config.gamma1*info_nce_loss + cross_entropy_loss
        labels, predictions = labels.cpu().detach(), predictions.cpu().detach()
        acc = metrics.accuracy_score(labels, predictions)
    loss.backward()
    optimizer.step()
    return loss, acc

def test(test_mask):
    model.eval()
    if config.method != "ConvCurv":
        logits, Reg1, Reg2 = model(data)
    else:
        _, embeddings = model(test_dataset)
        q, k, v, pairs, labels, sim_pairs = get_qkv_dataset(embeddings, test_dataset.pos_edge_index_0, test_dataset.pos_edge_index_1, test_dataset.neg_edge_index_0, test_dataset.neg_edge_index_1, 1)
        AB_similarities, _ = simnet.similarity_head(sim_pairs)
        _, sim_predictions = torch.max(AB_similarities, 1)

        logits = model.fc(pairs)
        nll_loss = F.log_softmax(logits, dim=1)
        cross_entropy_loss = F.nll_loss(nll_loss, labels)
        _, predictions = torch.max(nll_loss.data, 1)
        info_nce_loss = loss_function(q, k, v)
        loss = config.gamma1*info_nce_loss + cross_entropy_loss
        labels, predictions = labels.cpu().detach(), predictions.cpu().detach()
        acc = metrics.accuracy_score(labels, predictions)
        precision = metrics.precision_score(labels, predictions)
        recall = metrics.recall_score(labels, predictions)
        f1 = metrics.f1_score(labels, predictions)
        auc = metrics.roc_auc_score(labels, predictions)

        sim_acc = metrics.accuracy_score(labels, sim_predictions)
        sim_precision = metrics.precision_score(labels, sim_predictions)
        sim_recall = metrics.recall_score(labels, sim_predictions)
        sim_f1 = metrics.f1_score(labels, sim_predictions)
        sim_auc = metrics.roc_auc_score(labels, sim_predictions)
        print(f"sim_acc : {sim_acc}, sim_precision : {sim_precision}, sim_recall : {sim_recall}, sim_f1 : {sim_f1}, sim_auc : {sim_auc}")
# accs = []
    # for mask in [train_mask, val_mask, test_mask]:
    #     pred = logits[mask].max(1)[1]
    #     #print(pred)
    #     acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
    #     accs.append(acc)
    # accs.append(F.nll_loss(logits[val_mask], data.y[val_mask]))
    # print(accs)
    return loss, acc, precision, recall, f1, auc

config = Config()
print(config.__dict__)
#load dataset
times = range(config.times)  #Todo:实验次数
epoch_num = config.epoch_num
wait_total = config.patience
pipelines = [config.method]
# d_names=['Cora','Citeseer','PubMed']
d_names = config.d_names
simnet = torch.load("saves/pretrained/CUB/h4f2_86.5.pth")
train_dataset, test_dataset = get_GCN_data("CUB")
print("loading data, done.")
for time in times:
    train_mask = train_dataset.train_mask.bool()
    # val_mask=data.val_mask.bool()
    test_mask = test_dataset.test_mask.bool()
    model, train_dataset, test_dataset = ConvCurv.call(train_dataset, test_dataset, config.d_names, train_dataset.x.size(1), train_dataset.num_classes, config)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, weight_decay=0.0005)
    best_val_loss = np.inf
    for epoch in range(0, epoch_num):
        train_loss, train_acc = train(train_mask)
        test_loss, test_acc, test_precision, test_recall, test_f1, test_auc = test(test_mask)
        print(f"epoch {epoch}: training acc: {train_acc}, training loss: {train_loss}, test_acc: {test_acc}, test_precision: {test_precision}, test_recall: {test_recall}, test_f1: {test_f1}, test_auc: {test_auc}, test_loss: {test_loss}")
        # if info_nce_loss <= best_val_loss:
        #     best_val_loss = info_nce_loss
        #     wait_step = 0
        # else:
        #     wait_step += 1
        #     if wait_step == wait_total:
        #         print('Early stop! Min loss: ', best_val_loss)
        #         break