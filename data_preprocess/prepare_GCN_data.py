from torch_geometric.data import Data
from itertools import repeat
import torch
import pickle
from data_preprocess.preprocess import *
from torch_scatter import scatter

def edge_index_from_dict(graph_dict):
    row, col = [], []
    for key, value in graph_dict.items():
        row += repeat(key, len(value))
        col += value
    edge_index = torch.stack([torch.tensor(row), torch.tensor(col)], dim=0)
    return edge_index


def index_to_mask(index, size):
    mask = torch.zeros((size, ), dtype=torch.bool)
    mask[index] = 1
    return mask

def get_GCN_data(dataset_str):
    if dataset_str == "CUB":
        cos_str = "cos_mean_0.7"
    elif dataset_str == "Car":
        cos_str = "cos_mean_0.75"
    elif dataset_str == "Air":
        cos_str = "cos_mean_0.75"
    x_train, x_test, y_train, y_test = get_dataset(f"image_embeddings/{dataset_str}/ori_pretrained")
    edge_index_train, edge_weight_train, edge_curvature_train = get_edge_weight_curvature(dataset_str, weight_file_name=f"{cos_str}/base_train_subgraph", curvature_file_name=f"{cos_str}/graph_base_train.edge_list_OllivierRicci")
    edge_index_test, edge_weight_test, edge_curvature_test = get_edge_weight_curvature(dataset_str, weight_file_name=f"{cos_str}/base_test_subgraph", curvature_file_name=f"{cos_str}/graph_base_test.edge_list_OllivierRicci")
    pos_neg_edge_train = pickle.load(open(f"curvature/{dataset_str}/{cos_str}/base_train_pos_neg_edge.pkl", "rb"))
    pos_neg_edge_test = pickle.load(open(f"curvature/{dataset_str}/{cos_str}/base_test_pos_neg_edge.pkl", "rb"))

    # node_curvature_train = scatter(torch.tensor(edge_curvature_train), edge_index_train[0], dim=0, reduce="mean")
    # node_curvature_test = scatter(torch.tensor(edge_curvature_test), edge_index_test[0], dim=0, reduce="mean")

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

    train_index = torch.arange(y_train.size(0), dtype=torch.long)
    # val_index = torch.arange(y.size(0), y.size(0) + 500, dtype=torch.long)
    test_index = torch.arange(y_test.size(0), dtype=torch.long)
    train_mask = index_to_mask(train_index, size=y_train.size(0))
    # val_mask = index_to_mask(val_index, size=y.size(0))
    test_mask = index_to_mask(test_index, size=y_test.size(0))

    train_data = Data(x=x_train, edge_index=edge_index_train, y=y_train)
    # train_data.node_curvature = node_curvature_train.view(-1, 1)
    train_data.train_mask = train_mask
    train_data.edge_weight = edge_weight_train
    train_data.edge_curvature = edge_curvature_train
    train_data.num_classes = torch.max(y_train)+1
    train_data.pos_edge_index_0 = pos_edge_train[:, 0]
    train_data.pos_edge_index_1 = pos_edge_train[:, 1]
    train_data.neg_edge_index_0 = neg_edge_train[:, 0]
    train_data.neg_edge_index_1 = neg_edge_train[:, 1]

    test_data = Data(x=x_test, edge_index=edge_index_test, y=y_test)
    # test_data.node_curvature = node_curvature_test.view(-1, 1)
    test_data.test_mask = test_mask
    test_data.edge_weight = edge_weight_test
    test_data.edge_curvature = edge_curvature_test
    test_data.num_classes = torch.max(y_test)+1
    test_data.pos_edge_index_0 = pos_edge_test[:, 0]
    test_data.pos_edge_index_1 = pos_edge_test[:, 1]
    test_data.neg_edge_index_0 = neg_edge_test[:, 0]
    test_data.neg_edge_index_1 = neg_edge_test[:, 1]


    return train_data, test_data

def get_GCN_novel_data(dataset_str, category):
    if dataset_str == "CUB":
        cos_str = "cos_mean_0.7"
    elif dataset_str == "Car":
        cos_str = "cos_mean_0.75"
    elif dataset_str == "Air":
        cos_str = "cos_mean_0.75"

    dir_name = f"image_embeddings/{dataset_str}/ori_pretrained/{category}"
    x, y = get_novel_dataset(dir_name)
    edge_index, edge_weight, edge_curvature = get_edge_weight_curvature(dataset_str, weight_file_name=f"{cos_str}/novel/{category}_subgraph", curvature_file_name=f"{cos_str}/novel/graph_{category}.edge_list_OllivierRicci")
    pos_neg_edge = pickle.load(open(f"curvature/{dataset_str}/{cos_str}/novel/{category}_pos_neg_edge.pkl", "rb"))

    pos_edge = []
    neg_edge = []
    for key in pos_neg_edge["neg_edge"]:
        if key in pos_neg_edge["pos_edge"]:
            pos_edge += pos_neg_edge["pos_edge"][key]
            neg_edge += pos_neg_edge["neg_edge"][key]
    pos_edge = torch.tensor(pos_edge)
    neg_edge = torch.tensor(neg_edge)

    train_index = torch.arange(y.size(0), dtype=torch.long)
    train_mask = index_to_mask(train_index, size=y.size(0))

    data = Data(x=x, edge_index=edge_index, y=y)
    data.train_mask = train_mask
    data.edge_weight = edge_weight
    data.edge_curvature = edge_curvature
    data.num_classes = torch.max(y)+1
    data.pos_edge_index_0 = pos_edge[:, 0]
    data.pos_edge_index_1 = pos_edge[:, 1]
    data.neg_edge_index_0 = neg_edge[:, 0]
    data.neg_edge_index_1 = neg_edge[:, 1]


    return data

# train_data, test_data = get_GCN_data("CUB")
