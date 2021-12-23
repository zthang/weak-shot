import os
import pickle
import sys
from os.path import dirname, abspath
path = dirname(dirname(abspath(__file__)))
sys.path.append(path)
import numpy as np
from sklearn import metrics
from data_preprocess.preprocess import get_image_label
from tqdm import tqdm

lines = open("../graph/CUB/base_train_graph_mean_ori_pretrained.txt", "r").readlines()
base_train_label = get_image_label(f"../image_embeddings/CUB/ori_pretrained/image_category_base_train.pkl")
edge_list = []
for line in lines:
    data = line.strip().split(" ")
    edge_list.append([int(data[0]), int(data[1]), float(data[2])])
print("begin sorting...")
edge_list = sorted(edge_list, key=lambda x: x[2], reverse=True)
ground_truth = []
print("make label...")
for edge in edge_list:
    if base_train_label[edge[0]] != base_train_label[edge[1]]:
        ground_truth.append(0)
    else:
        ground_truth.append(1)
num = len(ground_truth)
accs = []
for i in tqdm(range(0, num, 1000)):
    pos = np.ones(i)
    neg = np.zeros(num-i)
    pred = np.concatenate((pos, neg), axis=0)
    acc = metrics.accuracy_score(ground_truth, pred)
    accs.append(acc)
pickle.dump(accs, open("saves/cos_distribute.pkl", "wb"))
print(1)