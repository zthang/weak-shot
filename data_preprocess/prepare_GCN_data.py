from torch_geometric.data import Data
from itertools import repeat
import torch
import pickle
from preprocess import get_dataset, get_edge_weight_curvature

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
        x_train, x_test, y_train, y_test = get_dataset("../image_embeddings/CUB/novel_train_pretrained")
        edge_index_train, edge_weight_train, edge_curvature_train = get_edge_weight_curvature("CUB", weight_file_name="cos_mean_0.77/base_train_subgraph", curvature_file_name="cos_mean_0.77/graph_base_train.edge_list_OllivierRicci")
        edge_index_test, edge_weight_test, edge_curvature_test = get_edge_weight_curvature("CUB", weight_file_name="cos_mean_0.77/base_test_subgraph", curvature_file_name="cos_mean_0.77/graph_base_test.edge_list_OllivierRicci")

        train_index = torch.arange(y_train.size(0), dtype=torch.long)
        # val_index = torch.arange(y.size(0), y.size(0) + 500, dtype=torch.long)
        test_index = torch.arange(y_test.size(0), dtype=torch.long)
        train_mask = index_to_mask(train_index, size=y_train.size(0))
        # val_mask = index_to_mask(val_index, size=y.size(0))
        test_mask = index_to_mask(test_index, size=y_test.size(0))

        train_data = Data(x=x_train, edge_index=edge_index_train, y=y_train)
        train_data.train_mask = train_mask
        train_data.edge_weight = edge_weight_train
        train_data.edge_curvature = edge_curvature_train
        train_data.num_classes = torch.max(y_train)+1

        test_data = Data(x=x_test, edge_index=edge_index_test, y=y_test)
        test_data.test_mask = test_mask
        test_data.edge_weight = edge_weight_test
        test_data.edge_curvature = edge_curvature_test
        test_data.num_classes = torch.max(y_test)+1

    return train_data, test_data



# train_data, test_data = get_GCN_data("CUB")
