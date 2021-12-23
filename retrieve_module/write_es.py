import os
from evaluate_ori import get_arg
from data.factory import get_data_helper
from tqdm import tqdm
from ad_similarity.ad_modules import *
import pickle
from elasticsearch import Elasticsearch
from elasticsearch import helpers
import time
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES']='1,2'

def write_es(name, embedding, label, index_name):
    hosts = [
        '10.10.10.10:9200',
    ]
    es = Elasticsearch(hosts)
    # helpers.bulk(es, insert_list)

    num_image = len(name)
    insert_list = []
    for i in range(num_image):
        json_data = {}
        json_data["embedding"] = embedding[i]
        json_data["label"] = label[i]
        json_data["name"] = name[i]
        insert_list.append({"_index": index_name,
                            "_type": "_doc",
                            "_source": json_data})
    helpers.bulk(es, insert_list, request_timeout=30)

args = get_arg()
simnet = GANSimilarityNet(args).cuda()
data_helper = get_data_helper(args)
noisy_novel_loader = data_helper.get_noisy_novel_loader()
for batch_i, image_info in tqdm(enumerate(noisy_novel_loader)):
    images, categories, file_names = image_info
    images = simnet.backbone(images)
    write_es(file_names, images.cpu().detach().numpy(), categories.numpy(), "web_noisy_novel_image")
