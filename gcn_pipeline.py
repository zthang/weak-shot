import os.path as osp
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
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
from data.factory import get_data_helper
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from imblearn.under_sampling import RandomUnderSampler
from torch.utils.data import Dataset, DataLoader
from evaluate_ori import get_arg
from ad_similarity.ad_modules import *
from utils.meters import MetrixMeter, AverageMeter
from tqdm import tqdm


τ = 0.5
class PairData(Dataset):
    def __init__(self, pair_index_0, pair_index_1):
        self.pair_index_0 = pair_index_0
        self.pair_index_1 = pair_index_1

    def __len__(self):
        return len(self.pair_index_0)

    def __getitem__(self, index):
        return [self.pair_index_0[index], self.pair_index_1[index]]

def get_qkv_dataset(embeddings, pos_edge_index_0, pos_edge_index_1, neg_edge_index_0, neg_edge_index_1, mode=0):
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
    pair_index_0 = torch.cat((pos_edge_index_0, neg_edge_index_0), dim=0)
    pair_index_1 = torch.cat((pos_edge_index_1, neg_edge_index_1), dim=0)
    if mode == 0:
        rus = RandomUnderSampler(random_state=0)
        index, labels = rus.fit_resample(torch.arange(labels.size(0)).view(-1, 1), labels)
        index = index.reshape(-1)
        pair_index_0 = pair_index_0[index]
        pair_index_1 = pair_index_1[index]

    loader = DataLoader(PairData(pair_index_0, pair_index_1), batch_size=10000, shuffle=(mode == 0), num_workers=0)
    return q, k, v, loader, torch.as_tensor(labels).cuda()


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
        q, k, v, pair_data_loader, labels = get_qkv_dataset(embeddings, train_dataset.pos_edge_index_0, train_dataset.pos_edge_index_1, train_dataset.neg_edge_index_0, train_dataset.neg_edge_index_1, 0)

        # main_meter = MetrixMeter(['Dissimilarity', 'Similarity'], default_metric='f1score')
        # cls_loss_avg = AverageMeter('Training cls loss')
        # pair_data_loader = A_pairs_data_loader
        # cross_entropy_loss = None
        # predictions = None
        # labels = None
        # for batch_i, A_data in tqdm(enumerate(pair_data_loader)):
        #     A_images, A_categories, A_names = A_data
        #     image_index = list(map(lambda x: name2index_train[x[0]], A_names))
        #     image_index = torch.tensor(image_index).view(-1, 1)
        #     index_0 = image_index.repeat(image_index.size(0), 1)
        #     index_1 = image_index.repeat(1, image_index.size(0)).view(-1, image_index.size(1))
        #     index_0 = index_0.view(-1).cuda()
        #     index_1 = index_1.view(-1).cuda()
        #     pair_0 = torch.index_select(embeddings, 0, index_0)
        #     pair_1 = torch.index_select(embeddings, 0, index_1)
        #     pairs = torch.cat((pair_0, pair_1), dim=1)
        #     logits = model.fc(pairs)
        #     _, pred = torch.max(logits, 1)
        #     targets = make_similarities(A_categories[0].cuda())
        #     main_meter.update(logits, targets)
        #     cls_loss = criterion(logits, targets)
        #     cls_loss_avg.update(cls_loss.mean().item())
        #     if batch_i != 0:
        #         cross_entropy_loss += cls_loss
        #         predictions = torch.cat((predictions, pred), dim=0)
        #         labels = torch.cat((labels, targets))
        #     else:
        #         cross_entropy_loss = cls_loss
        #         predictions = pred
        #         labels = targets
        # print(main_meter.report())

        logits = None
        for i, (pair_index_0, pair_index_1) in enumerate(pair_data_loader):
            pair_embedding_0 = torch.index_select(embeddings, 0, pair_index_0)
            pair_embedding_1 = torch.index_select(embeddings, 0, pair_index_1)
            pairs = torch.cat((pair_embedding_0, pair_embedding_1), dim=1)
            out = model.fc(pairs)
            if logits != None:
                logits = torch.cat((logits, out), dim=0)
            else:
                logits = out
        nll_loss = F.log_softmax(logits, dim=1)
        cross_entropy_loss = F.nll_loss(nll_loss, labels)
        _, predictions = torch.max(nll_loss.data, 1)

        info_nce_loss = loss_function(q, k, v)
        # loss = config.gamma1*info_nce_loss + cross_entropy_loss/(batch_i+1)
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
        q, k, v, pair_data_loader, labels = get_qkv_dataset(embeddings, test_dataset.pos_edge_index_0, test_dataset.pos_edge_index_1, test_dataset.neg_edge_index_0, test_dataset.neg_edge_index_1, 1)

        # meter = MetrixMeter(['Dissimilarity', 'Similarity'], default_metric='f1score')
        # pair_data_loader = base_similarity_test_loader
        # cross_entropy_loss = None
        # predictions = None
        # labels = None
        # for batch_i, A_data in tqdm(enumerate(pair_data_loader)):
        #     A_images, A_categories, A_names = A_data
        #     image_index = list(map(lambda x: name2index_test[x], A_names))
        #     image_index = torch.tensor(image_index).view(-1, 1)
        #     index_0 = image_index.repeat(image_index.size(0), 1)
        #     index_1 = image_index.repeat(1, image_index.size(0)).view(-1, image_index.size(1))
        #     index_0 = index_0.view(-1).cuda()
        #     index_1 = index_1.view(-1).cuda()
        #     pair_0 = torch.index_select(embeddings, 0, index_0)
        #     pair_1 = torch.index_select(embeddings, 0, index_1)
        #     pairs = torch.cat((pair_0, pair_1), dim=1)
        #     logits = model.fc(pairs)
        #     _, pred = torch.max(logits, 1)
        #     targets = make_similarities(A_categories.cuda())
        #     meter.update(logits, targets)
        #     cls_loss = criterion(logits, targets)
        #     if batch_i != 0:
        #         cross_entropy_loss += cls_loss
        #         predictions = torch.cat((predictions, pred), dim=0)
        #         labels = torch.cat((labels, targets))
        #     else:
        #         cross_entropy_loss = cls_loss
        #         predictions = pred
        #         labels = targets
        # print(meter.report())

        logits = None
        for i, (pair_index_0, pair_index_1) in enumerate(pair_data_loader):
            pair_embedding_0 = torch.index_select(embeddings, 0, pair_index_0)
            pair_embedding_1 = torch.index_select(embeddings, 0, pair_index_1)
            pairs = torch.cat((pair_embedding_0, pair_embedding_1), dim=1)
            out = model.fc(pairs)
            if logits != None:
                logits = torch.cat((logits, out), dim=0)
            else:
                logits = out
        nll_loss = F.log_softmax(logits, dim=1)
        cross_entropy_loss = F.nll_loss(nll_loss, labels)
        _, predictions = torch.max(nll_loss.data, 1)

        info_nce_loss = loss_function(q, k, v)
        # loss = config.gamma1*info_nce_loss + cross_entropy_loss/(batch_i+1)
        loss = config.gamma1*info_nce_loss + cross_entropy_loss
        labels, predictions = labels.cpu().detach(), predictions.cpu().detach()
        acc = metrics.accuracy_score(labels, predictions)
        precision = metrics.precision_score(labels, predictions)
        recall = metrics.recall_score(labels, predictions)
        f1 = metrics.f1_score(labels, predictions)
        auc = metrics.roc_auc_score(labels, predictions)
    return loss, acc, precision, recall, f1, auc, embeddings

if __name__ == '__main__':
    config = Config()
    print(config.__dict__)
    #load dataset
    times = range(config.times)  #Todo:实验次数
    epoch_num = config.epoch_num
    wait_total = config.patience
    pipelines = [config.method]
    # d_names=['Cora','Citeseer','PubMed']
    dataset_str = config.d_names
    train_dataset, test_dataset = get_GCN_data(dataset_str)
    print("loading data, done.")

    # args = get_arg()
    # data_helper = get_data_helper(args)
    # base_test_loader = data_helper.get_base_test_loader()
    # A_loader = data_helper.get_clean_base_loader()
    # B_loader = data_helper.get_noisy_novel_loader()
    # A_pairs_data_loader = get_train_loader_(args, A_loader, B_loader, args.batch_class_num)
    # base_similarity_test_loader = get_similarity_test_loader2(args, base_test_loader)
    # criterion = nn.CrossEntropyLoss().cuda()
    # name2index_train = {}
    # for index, image_name in enumerate(A_loader.dataset.image_list):
    #     name2index_train[image_name[0]] = index
    # name2index_test = {}
    # for index, image_name in enumerate(base_similarity_test_loader.dataset.image_list):
    #     name2index_test[image_name[0]] = index

    for time in times:
        train_mask = train_dataset.train_mask.bool()
        # val_mask=data.val_mask.bool()
        test_mask = test_dataset.test_mask.bool()
        model, train_dataset, test_dataset = ConvCurv.call(train_dataset, test_dataset, config.d_names, train_dataset.x.size(1), train_dataset.num_classes, config)
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, weight_decay=0.0005)
        best_val_loss = np.inf
        best_val_auc = -1
        save_embedding = None
        for epoch in range(0, epoch_num):
            train_loss, train_acc = train(train_mask)
            test_loss, test_acc, test_precision, test_recall, test_f1, test_auc, embeddings = test(test_mask)
            print(f"epoch {epoch}: training acc: {train_acc}, training loss: {train_loss}, test_acc: {test_acc}, test_precision: {test_precision}, test_recall: {test_recall}, test_f1: {test_f1}, test_auc: {test_auc}, test_loss: {test_loss}")
            if test_auc >= best_val_auc:
                best_val_auc = test_auc
                wait_step = 0
                if best_val_auc > 0.6:
                    current_time = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
                    torch.save(model, f"saves/model/curvGN_{dataset_str}_{current_time}_{best_val_auc:.4f}.pth")
                    print(f"save name: curvGN_{dataset_str}_{current_time}_{best_val_auc:.4f}.pth")
            else:
                wait_step += 1
                if wait_step == wait_total:
                    print('Early stop! Min loss: ', best_val_loss)
                    break