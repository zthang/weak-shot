import networkx as nx
import matplotlib.pyplot as plt
import pickle


lines = open("../graph/base_train_graph_1212.txt", "r").readlines()
edge_weight = [i.strip().split(' ') for i in lines]
base_train_label = pickle.load(open("../image_embeddings/image_category_base_train.pkl", "rb"))[:300]
labels = base_train_label.numpy()
print("data prepare done.")
G = nx.Graph()
for edge in edge_weight:
    G.add_edge(int(edge[0]), int(edge[1]), weight=edge[2])
# for u,v,d in G.edges(data=True):
#   print(u,v,d['weight'])
edge_labels = dict([((u, v,), d['weight']) for u, v, d in G.edges(data=True)])
print("edge init done.")
pos = nx.spring_layout(G)
nx.draw_networkx(G, pos=pos, nodelist=range(labels.shape[0]), node_color=labels)#绘制图中边的权重
print("draw done.")
nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)
print("draw_networkx_edge_labels done.")
# print(edge_labels)
# nx.draw_networkx(G)
plt.show()
# plt.savefig(f'../picture/base_train.png', bbox_inches='tight', dpi=255, pad_inches=0.0)