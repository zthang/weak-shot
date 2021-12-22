import os

import torch

os.environ['CUDA_VISIBLE_DEVICES']='2, 3'
from data_preprocess.preprocess import *
from torch.utils.data import Dataset, DataLoader
from imblearn.under_sampling import RandomUnderSampler
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from data.factory import get_data_helper
from utils.parser import get_base_parser, add_base_train
from ad_similarity.ad_modules import *
from utils.meters import MetrixMeter, AverageMeter
from utils.project_kits import init_log, log, set_seeds, occupy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def add_similarity_train(parser):
    parser.add_argument('--batch_class_num', default=20, type=int)
    parser.add_argument('--batch_class_num_B', default=1, type=int)
    parser.add_argument('--target_domain', default='novel_train', type=str)

    parser.add_argument('--beta', default=0, type=float)
    parser.add_argument('--domain_lr', default=0.01, type=float)

    parser.add_argument('--head_type', default='2', type=str)
    parser.add_argument('--similarity_weighted', default=0, type=int)

    parser.add_argument('--domain_bn', default=1, type=int)

    parser.add_argument('--source_set', default='clean_base', type=str)
    parser.add_argument('--target_set', default='noisy_novel', type=str)

    parser.add_argument('--noisy_frac', default=0, type=float)

    parser.add_argument('--similarity_pretrained', type=str, default='../saves/naive_NoisyNovel_CUB_lr0.0001_b64_wd0.0001_10280027/naive_NoisyNovel_CUB_lr0.0001_b64_wd0.0001_10280027_best.pth')
    return parser

def get_arg():
    parser = get_base_parser()
    parser = add_base_train(parser)
    parser = add_similarity_train(parser)
    args = parser.parse_args()

    args.beta = 0.
    args.lr = 5e-3
    args.domain_lr = 5e-3
    args.batch_size = 50
    args.lr_interval = 25
    args.num_epoch = 300
    args.head_type = 4
    args.batch_class_num_B = 2
    args.similarity_pretrained = "../saves/pretrained/CUB/pretrained_84.5.pth"
    return args

class TempData(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return None
class PairData(Dataset):
    def __init__(self, index_0, index_1, y):
        self.index_0 = index_0
        self.index_1 = index_1
        self.y = y
    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return [self.index_0[index], self.index_1[index], self.y[index]]


def test_1():
    #torch1.0.0

    x_train, x_test, y_train, y_test = get_dataset("image_embeddings/CUB/ori_pretrained")
    pos_neg_edge_train = pickle.load(open(f"curvature/CUB/cos_mean_0.7/base_train_pos_neg_edge.pkl", "rb"))
    pos_neg_edge_test = pickle.load(open(f"curvature/CUB/cos_mean_0.7/base_test_pos_neg_edge.pkl", "rb"))

    pos_edge_train = []
    neg_edge_train = []
    for key in pos_neg_edge_train["neg_edge"]:
        if key in pos_neg_edge_train["pos_edge"]:
            pos_edge_train += pos_neg_edge_train["pos_edge"][key]
            neg_edge_train += pos_neg_edge_train["neg_edge"][key]
    pos_edge_train = torch.tensor(pos_edge_train)
    neg_edge_train = torch.tensor(neg_edge_train)

    pos_edge_test = []
    neg_edge_test = []
    for key in pos_neg_edge_test["neg_edge"]:
        if key in pos_neg_edge_test["pos_edge"]:
            pos_edge_test += pos_neg_edge_test["pos_edge"][key]
            neg_edge_test += pos_neg_edge_test["neg_edge"][key]
    pos_edge_test = torch.tensor(pos_edge_test)
    neg_edge_test = torch.tensor(neg_edge_test)

    train_data = TempData(x=x_train, y=y_train)
    train_data.pos_edge_index_0 = pos_edge_train[:, 0]
    train_data.pos_edge_index_1 = pos_edge_train[:, 1]
    train_data.neg_edge_index_0 = neg_edge_train[:, 0]
    train_data.neg_edge_index_1 = neg_edge_train[:, 1]

    test_data = TempData(x=x_test, y=y_test)
    test_data.pos_edge_index_0 = pos_edge_test[:, 0].cuda()
    test_data.pos_edge_index_1 = pos_edge_test[:, 1].cuda()
    test_data.neg_edge_index_0 = neg_edge_test[:, 0].cuda()
    test_data.neg_edge_index_1 = neg_edge_test[:, 1].cuda()


    simnet = torch.load("saves/pretrained/CUB/h4f2_86.5.pth")
    simnet.reset_gpu()
    # rus = RandomUnderSampler(random_state=0)
    pos_labels = torch.ones(test_data.pos_edge_index_0.size(0), dtype=torch.long)
    neg_labels = torch.zeros(test_data.neg_edge_index_0.size(0), dtype=torch.long)
    labels = torch.cat((pos_labels, neg_labels), dim=0)
    # index, labels = rus.fit_resample(torch.arange(labels.size(0)).view(-1, 1), labels)
    # index = index.reshape(-1)

    index_0 = torch.cat((test_data.pos_edge_index_0, test_data.neg_edge_index_0), dim=0)
    index_1 = torch.cat((test_data.pos_edge_index_1, test_data.neg_edge_index_1), dim=0)
    # index_0 = index_0[index]
    # index_1 = index_1[index]
    eva_data = PairData(index_0, index_1, labels)
    train_loader = DataLoader(eva_data, batch_size=1000, shuffle=False, num_workers=0)
    pred = None
    for i, (sim_index_0, sim_index_1, y) in enumerate(train_loader):
        print(f"{i*1000} pairs done.")
        sim_index_0 = torch.index_select(test_data.x, 0, sim_index_0)
        sim_index_1 = torch.index_select(test_data.x, 0, sim_index_1)
        sim_pairs = torch.cat((sim_index_0, sim_index_1), dim=1)
        AB_similarities, _ = simnet.similarity_head(sim_pairs)
        _, sim_predictions = torch.max(AB_similarities, 1)
        if i != 0:
            pred = torch.cat((pred, sim_predictions), dim=0)
        else:
            pred = sim_predictions

    pred = pred.cpu().detach()
    acc = metrics.accuracy_score(labels, pred)
    precision = metrics.precision_score(labels, pred)
    recall = metrics.recall_score(labels, pred)
    f1 = metrics.f1_score(labels, pred)
    auc = metrics.roc_auc_score(labels, pred)
    print(f"acc: {acc:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}, auc: {auc:.4f}")

def test_2():
    from CurvGN_module import ConvCurv
    from data_preprocess.prepare_GCN_data import get_GCN_data
    train_dataset, test_dataset = get_GCN_data("CUB")
    print("loading data, done.")
    _, train_dataset, test_dataset = ConvCurv.call(train_dataset, test_dataset, "CUB", train_dataset.x.size(1), train_dataset.num_classes, None)
    args = get_arg()
    data_helper = get_data_helper(args)
    base_test_loader = data_helper.get_base_test_loader()
    base_similarity_test_loader = get_similarity_test_loader2(args, base_test_loader)
    meter = MetrixMeter(['Dissimilarity', 'Similarity'], default_metric='f1score')
    name2index = {}
    model = torch.load("saves/model/curvGN_1218_0.6292.pth")
    print("loading model, done.")
    _, embeddings = model(test_dataset)
    for index, image_name in enumerate(base_similarity_test_loader.dataset.image_list):
        name2index[image_name[0]] = index
    for batch_i, (images, categories, im_names) in tqdm(enumerate(base_similarity_test_loader)):
        image_index = list(map(lambda x: name2index[x], im_names))
        # image_index = torch.arange(0, 60).view(-1, 1).cuda()
        image_index = torch.tensor(image_index).view(-1, 1)
        index_0 = image_index.repeat(image_index.size(0), 1)
        index_1 = image_index.repeat(1, image_index.size(0)).view(-1, image_index.size(1))
        index_0 = index_0.view(-1).cuda()
        index_1 = index_1.view(-1).cuda()
        pair_0 = torch.index_select(embeddings, 0, index_0)
        pair_1 = torch.index_select(embeddings, 0, index_1)
        pairs = torch.cat((pair_0, pair_1), dim=1)
        logits = model.fc(pairs)
        # _, pred = torch.max(logits, 1)
        targets = make_similarities(categories)
        meter.update(logits, targets)
    print(meter.report())

def test_3():
    from data_preprocess.prepare_GCN_data import get_GCN_data
    from gcn_pipeline import get_qkv_dataset
    from CurvGN_module import ConvCurv
    train_dataset, test_dataset = get_GCN_data("CUB")
    print("loading data, done.")
    _, train_dataset, test_dataset = ConvCurv.call(train_dataset, test_dataset, "CUB", train_dataset.x.size(1), train_dataset.num_classes, None)
    model = torch.load("saves/model/curvGN_1218_0.6246.pth")
    print("loading model, done.")
    _, embeddings = model(test_dataset)
    q, k, v, pair_data_loader, labels = get_qkv_dataset(embeddings, test_dataset.pos_edge_index_0, test_dataset.pos_edge_index_1, test_dataset.neg_edge_index_0, test_dataset.neg_edge_index_1, 1)
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
    predictions = predictions.cpu()
    labels = labels.cpu()
    acc = metrics.accuracy_score(labels, predictions)
    precision = metrics.precision_score(labels, predictions)
    recall = metrics.recall_score(labels, predictions)
    f1 = metrics.f1_score(labels, predictions)
    auc = metrics.roc_auc_score(labels, predictions)
    print(acc, precision, recall, f1, auc)

if __name__ == '__main__':
    test_3()