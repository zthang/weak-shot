import torch

from data_preprocess.prepare_GCN_data import *
from CurvGN_module import ConvCurv
from torch_geometric.utils import add_self_loops, remove_self_loops,degree,softmax
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
class PairData(Dataset):
    def __init__(self, index_0, index_1):
        self.index_0 = index_0.to(device)
        self.index_1 = index_1.to(device)
    def __len__(self):
        return len(self.index_0)

    def __getitem__(self, index):
        return [self.index_0[index], self.index_1[index]]

def minmaxscaler(data):
    min = torch.min(data)
    max = torch.max(data)
    return (data - min)/(max-min)

cub_novel_category = ['043.Yellow_bellied_Flycatcher',
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

def calcu_cub_weight():
    model = torch.load("saves/model/curvGN_1218_0.6246.pth")
    print("loading model, done.")
    for idx, category in enumerate(cub_novel_category):
        print(category)
        data = get_GCN_novel_data("CUB", category)
        print("loading data, done.")
        begin_index = idx*30
        end_index = idx*30+1000
        if data.y[begin_index] != idx or data.y[end_index-1] != idx:
            raise ValueError
        edge_index_0 = []
        edge_index_1 = []
        for i in range(data.edge_index.size(1)):
            if data.edge_index[0][i] >= begin_index and data.edge_index[0][i] < end_index and data.edge_index[1][i] >= begin_index and data.edge_index[1][i] < end_index:
                edge_index_0.append(data.edge_index[0][i])
                edge_index_1.append(data.edge_index[1][i])
        print(f"edge num: {len(edge_index_0)}")
        pair_data_set = PairData(torch.tensor(edge_index_0), torch.tensor(edge_index_1))
        pair_data_loader = DataLoader(pair_data_set, batch_size=10000, shuffle=False, num_workers=0)
        edge_curvature = data.edge_curvature
        edge_weight = data.edge_weight
        edge_curvature = edge_curvature + [0 for i in range(data.x.size(0))]
        edge_weight = edge_weight + [1 for i in range(data.x.size(0))]
        edge_curvature = torch.tensor(edge_curvature, dtype=torch.float).view(-1, 1)
        edge_weight = torch.tensor(edge_weight, dtype=torch.float).view(-1, 1)
        data.edge_index, _ = remove_self_loops(data.edge_index)
        data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.x.size(0))
        data.w_mul = torch.cat((edge_curvature, edge_weight), dim=1)
        data.w_mul = data.w_mul.to(device)
        data = data.to(device)
        _, embeddings = model(data)
        logits = []
        for i, (pair_index_0, pair_index_1) in enumerate(pair_data_loader):
            pair_embedding_0 = torch.index_select(embeddings, 0, pair_index_0)
            pair_embedding_1 = torch.index_select(embeddings, 0, pair_index_1)
            pairs = torch.cat((pair_embedding_0, pair_embedding_1), dim=1)
            out = model.fc(pairs)
            logits.append(out)
        logits = torch.cat(logits)
        nll_loss = F.log_softmax(logits, dim=1)
        pred = torch.exp(nll_loss)
        node_index = pair_data_set.index_0 - begin_index
        node_weight = scatter(pred[:, 1], node_index, dim=0, reduce="mean", dim_size=1000)+0.1
        node_weight /= node_weight.mean()
        torch.save(node_weight, f"weight/CUB/{category}_weight_new.pth")
calcu_cub_weight()