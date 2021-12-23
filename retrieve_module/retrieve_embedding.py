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
# from evaluate_ori import get_arg
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

k = 50
# args = get_arg()
# args.batch_size = 1
# simnet = GANSimilarityNet(args).cuda()
# data_helper = get_data_helper(args)
# test_loader = data_helper.get_noisy_novel_loader()
retrieve_dict = {}
# test_image_name = []
test_image_name = pickle.load(open("../saves/CUB/novel_test_name_image.pkl", "rb"))
for batch_i, image_info in tqdm(enumerate(test_image_name)):
    # images, c, file_names = image_info
    file_name, image = image_info
    # images = simnet.backbone(images)
    # test_image_name.append((file_names[0].replace('\\', os.sep), images[0].cpu().detach().numpy()))
    retrieve_info = retrieve_image("web_noisy_novel_image", image, k)
    distribution = torch.zeros(50)
    for info in retrieve_info:
        distribution[info["_source"]["label"]] += 1*info["_score"]
    distribution = distribution/torch.sum(distribution)
    print(distribution.numpy())
    retrieve_dict[file_name] = distribution
torch.save(retrieve_dict, f"../weight/CUB/retrieve_noisy_dict_{k}_sum.pth")
print(f"../weight/CUB/retrieve_noisy_dict_{k}_sum.pth")
# pickle.dump(test_image_name, open("../saves/CUB/novel_train_name_image.pkl", "wb"))