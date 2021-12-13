from torch_geometric.data import Data
from itertools import repeat
import torch
import pickle

def edge_index_from_dict(graph_dict, num_nodes=None, coalesce=True):
    row, col = [], []
    for key, value in graph_dict.items():
        row += repeat(key, len(value))
        col += value
    edge_index = torch.stack([torch.tensor(row), torch.tensor(col)], dim=0)
    if coalesce:
        # NOTE: There are some duplicated edges and self loops in the datasets.
        #       Other implementations do not remove them!
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = coalesce_fn(edge_index, None, num_nodes, num_nodes)
    return edge_index


def index_to_mask(index, size):
    mask = torch.zeros((size, ), dtype=torch.bool)
    mask[index] = 1
    return mask


def get_GCN_data(dataset_str):
    if dataset_str == "CUB":
        train_index = torch.arange(y.size(0), dtype=torch.long)
