import pickle

import elasticsearch.helpers
import elasticsearch
import os
import sys
from os.path import dirname, abspath
path = dirname(dirname(abspath(__file__)))
sys.path.append(path)
import torch.nn.functional

os.environ['CUDA_VISIBLE_DEVICES']='2,3'
from evaluate_ori import get_arg
from data.factory import get_data_helper
from tqdm import tqdm
from ad_similarity.ad_modules import *

def retrieve_image(index, query_vector, k=30):
    hosts = [
        '10.10.10.10:9200',
    ]
    es = elasticsearch.Elasticsearch(hosts)
    searchbody ={
        "size" : k,
        "query": {
            "script_score": {
                "query": {
                    "match_all": {}
                },
                "script": {
                    "source": "cosineSimilarity(params.query_vector,doc['embedding']) + 1.0",
                    "params": {
                        "query_vector": query_vector

                    }
                }
            }
        }
    }
    embedding_info = es.search(index=index, body=searchbody, request_timeout=30).get('hits').get('hits')
    return embedding_info

def save_test_embedding(dataset_str):
    args = get_arg()
    args.batch_size = 1
    args.data_path = f'/home/zthang/zthang/SimTrans-Weak-Shot-Classification/workspace/dataset/{dataset_str}'
    if dataset_str == "Car":
        args.similarity_pretrained = "../saves/naive_NoisyNovel_Car_lr0.005_b192_wd0.0001_12230005/naive_NoisyNovel_Car_lr0.005_b192_wd0.0001_12230005_0.7989.pth"
    elif dataset_str == "Air":
        args.similarity_pretrained = "../saves/naive_NoisyNovel_Car_lr0.005_b192_wd0.0001_12230005/naive_NoisyNovel_Car_lr0.005_b192_wd0.0001_12230005_0.7989.pth"
    simnet = GANSimilarityNet(args).cuda()
    data_helper = get_data_helper(args)
    test_loader = data_helper.get_novel_test_loader()
    test_image_name = []
    for batch_i, image_info in tqdm(enumerate(test_loader)):
        images, c, file_names = image_info
        images = simnet.backbone(images)
        test_image_name.append((file_names[0].replace('\\', os.sep), images[0].cpu().detach().numpy()))
    pickle.dump(test_image_name, open(f"../saves/{dataset_str}/novel_test_name_image.pkl", "wb"))

def save_retrieve_file(dataset_str, index):
    k = 10
    retrieve_dict = {}
    if dataset_str == "Car":
        category_num = 108
    elif dataset_str == "Air":
        category_num = 25

    test_name_image = pickle.load(open(f"../saves/{dataset_str}/novel_test_name_image.pkl", "rb"))
    for batch_i, image_info in tqdm(enumerate(test_name_image)):
        file_name, image = image_info
        retrieve_info = retrieve_image(index, image, k)
        distribution = torch.zeros(category_num)
        for info in retrieve_info:
            distribution[info["_source"]["label"]] += 1*info["_score"]
        distribution = distribution/torch.sum(distribution)
        print(distribution.numpy())
        retrieve_dict[file_name] = distribution
    torch.save(retrieve_dict, f"../weight/{dataset_str}/retrieve_noisy_dict_{k}_sum.pth")
    print(f"../weight/{dataset_str}/retrieve_noisy_dict_{k}_sum.pth")

# save_test_embedding("Car")
# save_test_embedding("Air")

save_retrieve_file("Car", "web_noisy_novel_image_car")
save_retrieve_file("Air", "web_noisy_novel_image_air")