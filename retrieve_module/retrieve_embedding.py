import elasticsearch.helpers
import elasticsearch
import os

import torch.nn.functional

os.environ['CUDA_VISIBLE_DEVICES']='0, 1'
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

args = get_arg()
args.batch_size = 1
simnet = GANSimilarityNet(args).cuda()
data_helper = get_data_helper(args)
test_loader = data_helper.get_novel_test_loader()
retrieve_dict = {}
for batch_i, image_info in tqdm(enumerate(test_loader)):
    images, categories, file_names = image_info
    images = simnet.backbone(images)
    retrieve_info = retrieve_image("web_novel_image", images[0].cpu().detach().numpy())
    distribution = torch.zeros(50)
    for info in retrieve_info:
        distribution[info["_source"]["label"]] += 1
    distribution = torch.nn.functional.softmax(distribution)
    retrieve_dict[file_names[0].replace('\\', os.sep)] = distribution
    print(distribution.numpy())
torch.save(retrieve_dict, "../weight/CUB/retrieve_dict.pth")
