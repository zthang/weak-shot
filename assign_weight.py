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
air_novel_category = ['F/A-18',
                      'Eurofighter Typhoon',
                      'DHC-6',
                      'C-130',
                      'SR-20',
                      'Gulfstream V',
                      'DHC-1',
                      'CRJ-900',
                      '757-300',
                      'Fokker 70',
                      'PA-28',
                      'Saab 2000',
                      'Tu-154',
                      'Spitfire',
                      'Model B200',
                      'Saab 340',
                      'An-12',
                      'Cessna 208',
                      'A321',
                      'DR-400',
                      'Cessna 525',
                      'ATR-72',
                      'Gulfstream IV',
                      '737-200',
                      'Cessna 172']
car_novel_category = ['ASX abroad version',
                      'Alto',
                      'Audi A1',
                      'Audi A4L',
                      'Audi A6L',
                      'Audi Q5',
                      'Audi TT coupe',
                      'Aveo sedan',
                      'BWM 1 Series hatchback',
                      'BWM 3 Series convertible',
                      'BWM 5 Series',
                      'BWM M5',
                      'BWM X5',
                      'BYD F6',
                      'BYD S6',
                      'Bei Douxing',
                      'Benz A Class',
                      'Benz E Class',
                      'Benz GL Class',
                      'Benz SLK Class',
                      'Besturn X80',
                      'Brabus S Class',
                      'Cadillac ATS-L',
                      'Camaro',
                      'Captiva',
                      'Chrey A3 hatchback',
                      'Chrey QQ3',
                      'Citroen C2',
                      'Classic Focus hatchback',
                      'Compass',
                      'Cross Lavida',
                      'Cruze hatchback',
                      'DS 4',
                      'Discovery',
                      'Eastar Cross',
                      'Enclave',
                      'Evoque',
                      'Family M5',
                      'Fengshen H30',
                      'Focus ST',
                      'GTC hatchback',
                      'Geely EC8',
                      'Golf convertible',
                      'Grandtiger G3',
                      'Great Wall M4',
                      'Haima S7',
                      'Haydo',
                      'Huaguan',
                      'Infiniti Q50',
                      'Infiniti QX80',
                      'Jingyue',
                      'KIA K5',
                      'Koleos',
                      'Landwind X8',
                      'Lechi',
                      'Lexus GS',
                      'Lexus IS convertible',
                      'Lifan 320',
                      'Linian S1',
                      'MAXUS V80xs',
                      'MG6 hatchback',
                      'MINI CLUBMAN',
                      'Magotan',
                      'Mazda 2 sedan',
                      'Mazda 3 abroad version',
                      'Mazda CX7',
                      'Mitsubishi Lancer EX',
                      'New Focus hatchback',
                      'Nissan NV200',
                      'Panamera',
                      'Peugeot 2008',
                      'Peugeot 3008',
                      'Peugeot 308',
                      'Peugeot 408',
                      'Polo hatchback',
                      'Premacy',
                      'Qiteng M70',
                      'Quatre sedan',
                      'Regal',
                      'Roewe 350',
                      'Ruifeng M5',
                      'Ruiyi',
                      'SAAB D70',
                      'Sail sedan',
                      'Scirocco',
                      'Shuma',
                      'Soul',
                      'Sunshine',
                      'Teana',
                      'Tiggo',
                      'Tiida',
                      'Toyota 86',
                      'Veloster',
                      'Verna',
                      'Volvo C30',
                      'Volvo S60L',
                      'Volvo V60',
                      'Weizhi',
                      'Wingle 5',
                      'Wulingzhiguang',
                      'Yaris',
                      'Youyou',
                      'Yuexiang sedan',
                      'Zhixiang',
                      'Zhonghua H530',
                      'Zhonghua Junjie FSV',
                      'Ziyoujian',
                      'i30']

def calcu_weight(dataset_str):
    if dataset_str == "CUB":
        model = torch.load("saves/model/curvGN_1218_0.6246.pth")
        novel_category = cub_novel_category
    elif dataset_str == "Car":
        model = torch.load("saves/model/curvGN_Car_2021-12-23-19:37:15_0.5079.pth")
        novel_category = car_novel_category
    elif dataset_str == "Air":
        model = torch.load("saves/model/curvGN_Air_2021-12-24-11:09:55_0.6257.pth")
        novel_category = air_novel_category
    print("loading model, done.")
    for idx, category in enumerate(novel_category):
        print(category)
        data = get_GCN_novel_data(dataset_str, category)
        print("loading data, done.")
        end_index = len(data.y)
        for category_index, category_value in enumerate(data.y):
            if category_value == idx:
                begin_index = category_index
                break
        for category_index, category_value in enumerate(data.y):
            if category_value == idx+1:
                end_index = category_index
                break
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
        node_weight = scatter(pred[:, 1], node_index, dim=0, reduce="mean", dim_size=end_index-begin_index)+0.3
        node_weight /= node_weight.mean()
        torch.save(node_weight, f"weight/{dataset_str}/{category}_weight_0.6.pth")
calcu_weight("CUB")