import networkx as nx
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from GraphRicciCurvature.FormanRicci import FormanRicci
import pickle
import os
import time

def make_curvature_file(prefix, type, weight_threshold=0.77, method=0):
    lines = open(f"../graph/{type}_graph_mean.txt").readlines()
    dir_path = f"../{prefix}_{weight_threshold}"

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    f = open(f"{dir_path}/{type}_subgraph.txt", "w")
    Gd = nx.Graph()

    edge_weight = []
    for line in lines:
        data = line.strip().split(" ")
        edge_weight.append([int(data[0]), int(data[1]), float(data[2])])
    for edge in edge_weight:
        if edge[2] > weight_threshold:
            Gd.add_edge(edge[0], edge[1], weight=edge[2])
            f.write(f"{edge[0]} {edge[1]} {edge[2]}\n")
    f.close()
    current_time = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    print(f"{current_time} write subgraph done.")
    Gd_OT = OllivierRicci(Gd, weight="weight", alpha=0.5, method="OTD", verbose="INFO") if method == 0 else FormanRicci(Gd)
    Gd_OT.compute_ricci_curvature()
    print(f"{current_time} calculate curvature done.")
    with open(f"{dir_path}/graph_{type}.edge_list_OllivierRicci.txt" if method == 0 else f"{dir_path}/graph_{type}.edge_list_FormanRicci.txt", "w") as f:
        for item in Gd.edges:
            f.writelines(f"{item[0]} {item[1]} {Gd_OT.G[item[0]][item[1]]['ricciCurvature']}\n") if method == 0 else f.writelines(f"{item[0]} {item[1]} {Gd_OT.G[item[0]][item[1]]['formanCurvature']}\n")
        f.close()
    current_time = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    print(f"{current_time} done.")

# make_curvature_file("curvature/cos_mean", "base_train")
# lines = open("curvature/cos_mean_0.77/base_train_subgraph.txt", "r").readlines()
lines = open("../curvature/cos_mean_0.77/graph_base_train.edge_list_OllivierRicci.txt", "r").readlines()
f = open("../curvature/cos_mean_0.77/graph_base_train.edge_list_OllivierRicci_10class.txt", "w")
for line in lines:
    data = line.strip().split(" ")
    data[0] = int(data[0])
    data[1] = int(data[1])
    data[2] = float(data[2])
    if(data[0] < 300 and data[1] < 300):
        f.write(f"{data[0]} {data[1]} {data[2]}\n")
print(1)