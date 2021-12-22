import networkx as nx
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from GraphRicciCurvature.FormanRicci import FormanRicci
import pickle
import os
import time
import sys
from os.path import dirname, abspath
path = dirname(dirname(abspath(__file__)))
print(path)
sys.path.append(path)
from data_preprocess.preprocess import *

def make_curvature_file(datset_str, prefix, type, weight_threshold=0.70, method=0):
    # lines = open(f"../graph/{datset_str}/{type}_graph_mean_ori_pretrained.txt").readlines()
    # dir_path = f"../curvature/{datset_str}/{prefix}_{weight_threshold}"
    print(type)
    lines = open(f"../graph/{datset_str}/novel/{type}.txt").readlines()
    dir_path = f"../curvature/{datset_str}/{prefix}_{weight_threshold}/novel"

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    f = open(f"{dir_path}/{type}_subgraph.txt", "w")
    Gd = nx.Graph()
    labels = get_image_label(f"../image_embeddings/CUB/ori_pretrained/{type}/image_category_web.pkl").numpy()
    pos_edge = {}
    neg_edge = {}
    edge_weight = []
    for line in lines:
        data = line.strip().split(" ")
        edge_weight.append([int(data[0]), int(data[1]), float(data[2])])
    num = 0
    for edge in edge_weight:
        if edge[2] > weight_threshold:
            num += 1
            Gd.add_edge(edge[0], edge[1], weight=edge[2])
            f.write(f"{edge[0]} {edge[1]} {edge[2]}\n")
            if labels[edge[0]] != labels[edge[1]]:
                if edge[0] not in neg_edge:
                    neg_edge[edge[0]] = []
                neg_edge[edge[0]].append((edge[0], edge[1]))
                if edge[1] not in neg_edge:
                    neg_edge[edge[1]] = []
                neg_edge[edge[1]].append((edge[1], edge[0]))
            else:
                if edge[0] not in pos_edge:
                    pos_edge[edge[0]] = []
                pos_edge[edge[0]].append((edge[0], edge[1]))
                if edge[1] not in pos_edge:
                    pos_edge[edge[1]] = []
                pos_edge[edge[1]].append((edge[1], edge[0]))
    print(num)
    pickle.dump({
        "pos_edge": pos_edge,
        "neg_edge": neg_edge
    }, open(f"{dir_path}/{type}_pos_neg_edge.pkl", "wb"))
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


novel_category = ['043.Yellow_bellied_Flycatcher',
                  '111.Loggerhead_Shrike',
                  '023.Brandt_Cormorant',
                  '098.Scott_Oriole',
                  '055.Evening_Grosbeak',
                  '130.Tree_Sparrow',
                  '139.Scarlet_Tanager',
                  '123.Henslow_Sparrow',
                  '156.White_eyed_Vireo',
                  '124.Le_Conte_Sparrow',
                  '200.Common_Yellowthroat',
                  '072.Pomarine_Jaeger',
                  '173.Orange_crowned_Warbler',
                  '028.Brown_Creeper',
                  '119.Field_Sparrow',
                  '165.Chestnut_sided_Warbler',
                  '103.Sayornis',
                  '180.Wilson_Warbler',
                  '077.Tropical_Kingbird',
                  '012.Yellow_headed_Blackbird',
                  '045.Northern_Fulmar',
                  '190.Red_cockaded_Woodpecker',
                  '191.Red_headed_Woodpecker',
                  '138.Tree_Swallow',
                  '157.Yellow_throated_Vireo',
                  '052.Pied_billed_Grebe',
                  '033.Yellow_billed_Cuckoo',
                  '164.Cerulean_Warbler',
                  '031.Black_billed_Cuckoo',
                  '143.Caspian_Tern',
                  '094.White_breasted_Nuthatch',
                  '070.Green_Violetear',
                  '097.Orchard_Oriole',
                  '091.Mockingbird',
                  '104.American_Pipit',
                  '127.Savannah_Sparrow',
                  '161.Blue_winged_Warbler',
                  '049.Boat_tailed_Grackle',
                  '169.Magnolia_Warbler',
                  '148.Green_tailed_Towhee',
                  '113.Baird_Sparrow',
                  '087.Mallard',
                  '163.Cape_May_Warbler',
                  '136.Barn_Swallow',
                  '188.Pileated_Woodpecker',
                  '084.Red_legged_Kittiwake',
                  '026.Bronzed_Cowbird',
                  '004.Groove_billed_Ani',
                  '132.White_crowned_Sparrow',
                  '168.Kentucky_Warbler']

for c in novel_category[10:]:
    make_curvature_file("CUB", "cos_mean", c)
# make_curvature_file("CUB", "cos_mean", "base_train")
# lines = open("../graph/CUB/base_train_graph_mean_ori_pretrained.txt", "r").readlines()
# f = open("../graph/CUB/base_train_graph_mean_ori_pretrained_10class.txt", "w")

# lines = open("../curvature/CUB/cos_mean_0.7/graph_base_train.edge_list_OllivierRicci.txt", "r").readlines()
# f = open("../curvature/CUB/cos_mean_0.7/graph_base_train.edge_list_OllivierRicci_10class.txt", "w")
# valid_list = []
# def calcu_data(value):
#     if value >= 183 and value < 303:
#         value -= 123
#     elif value >= 459 and value < 490:
#         value -= 279
#     elif value >= 567 and value < 657:
#         value -= 357
#     return value
# for i in range(60):
#     valid_list.append(i)
# for i in range(183, 303):
#     valid_list.append(i)
# for i in range(459, 489):
#     valid_list.append(i)
# for i in range(567, 657):
#     valid_list.append(i)
# for line in lines:
#     data = line.strip().split(" ")
#     data[0] = int(data[0])
#     data[1] = int(data[1])
#     data[2] = float(data[2])
# # #     # if(data[0] in valid_list and data[1] in valid_list):
# # #     #     f.write(f"{calcu_data(data[0])} {calcu_data(data[1])} {data[2]}\n")
#     if(data[0] < 300 and data[1] < 300):
#         f.write(f"{data[0]} {data[1]} {data[2]}\n")
# print(1)