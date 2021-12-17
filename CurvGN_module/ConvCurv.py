import os.path as osp
import torch
import torch.nn.functional as F
from torch.nn import Linear
import torch_geometric.transforms as T
import numpy as np
import pickle
from torch_geometric.utils import add_self_loops, remove_self_loops,degree,softmax
from CurvGN_module.curvGN import curvGN
from torch_geometric.nn import GATConv, GCNConv

def minmaxscaler(x):
    for i in range(len(x)):
        min = np.amin(x[i])
        max = np.amax(x[i])
        x[i]=(x[i] - min)/(max-min)
    return x

class Net(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(Net, self).__init__()
        self.conv1 = curvGN(in_channels=num_features, out_channels=128)
        self.conv2 = curvGN(in_channels=128, out_channels=num_classes)
        # self.conv1 = GATConv(num_features, 64, heads=8, dropout=0.5)
        # self.conv2 = GATConv(64 * 8, num_classes, heads=1, concat=False, dropout=0.5)
        # self.conv1 = GCNConv(num_features, 256)
        # self.conv2 = GCNConv(256, num_classes)
        self.fc = Linear(2*(num_classes), 2)
    def forward(self, data):
        # x = torch.cat((data.x, data.node_curvature), dim=1)
        x = F.dropout(data.x, p=0.5, training=self.training)
        x = self.conv1(x, data.edge_index, data.w_mul)
        x = F.selu(x)
        # x = torch.cat((x, data.node_curvature), dim=1)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, data.edge_index, data.w_mul)
        # x = torch.cat((x, data.node_curvature), dim=1)
        # x_n = x.detach().numpy()
        # x_n = minmaxscaler(x_n)
        # with open("vectors_norm","wb") as f:
        #     pickle.dump(x_n,f)
        return F.log_softmax(x, dim=1), x

def num(strings):
    try:
        return int(strings)
    except ValueError:
        return float(strings)

def call(train_dataset, test_dataset, name, num_features, num_classes, config, hidden_state=128):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    edge_curvature_train = train_dataset.edge_curvature
    edge_weight_train = train_dataset.edge_weight
    edge_curvature_train = edge_curvature_train + [0 for i in range(train_dataset.x.size(0))]
    edge_weight_train = edge_weight_train + [1 for i in range(train_dataset.x.size(0))]
    edge_curvature_train = torch.tensor(edge_curvature_train, dtype=torch.float).view(-1, 1)
    edge_weight_train = torch.tensor(edge_weight_train, dtype=torch.float).view(-1, 1)
    train_dataset.edge_index, _ = remove_self_loops(train_dataset.edge_index)
    train_dataset.edge_index, _ = add_self_loops(train_dataset.edge_index, num_nodes=train_dataset.x.size(0))
    train_dataset.w_mul = torch.cat((edge_curvature_train, edge_weight_train), dim=1)
    train_dataset.w_mul = train_dataset.w_mul.to(device)
    edge_curvature_test = test_dataset.edge_curvature
    edge_weight_test = test_dataset.edge_weight
    edge_curvature_test = edge_curvature_test + [0 for i in range(test_dataset.x.size(0))]
    edge_weight_test = edge_weight_test + [1 for i in range(test_dataset.x.size(0))]
    edge_curvature_test = torch.tensor(edge_curvature_test, dtype=torch.float).view(-1, 1)
    edge_weight_test = torch.tensor(edge_weight_test, dtype=torch.float).view(-1, 1)
    test_dataset.edge_index, _ = remove_self_loops(test_dataset.edge_index)
    test_dataset.edge_index, _ = add_self_loops(test_dataset.edge_index, num_nodes=test_dataset.x.size(0))
    test_dataset.w_mul = torch.cat((edge_curvature_test, edge_weight_test), dim=1)
    test_dataset.w_mul = test_dataset.w_mul.to(device)
    model, train_dataset, test_dataset = Net(num_features, hidden_state).to(device), train_dataset.to(device), test_dataset.to(device)
    return model, train_dataset, test_dataset
