import pickle
import sys
import warnings
import os

warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from os.path import dirname, abspath
path = dirname(dirname(abspath(__file__)))
sys.path.append(path)
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm

import utils.saver as saver
from data.factory import get_data_helper

from utils.meters import MetrixMeter, AverageMeter
from utils.parser import get_base_parser, add_base_train
from utils.project_kits import init_log, log, set_seeds, occupy
from utils.vis import vis_acc
from ad_similarity.ad_modules import *
from data_preprocess.preprocess import *

def add_similarity_train(parser):
    parser.add_argument('--batch_class_num', default=20, type=int)
    parser.add_argument('--batch_class_num_B', default=1, type=int)
    parser.add_argument('--target_domain', default='novel_train', type=str)

    parser.add_argument('--beta', default=0, type=float)
    parser.add_argument('--domain_lr', default=0.01, type=float)

    parser.add_argument('--head_type', default='2', type=str)
    parser.add_argument('--similarity_weighted', default=0, type=int)

    parser.add_argument('--domain_bn', default=1, type=int)

    parser.add_argument('--source_set', default='clean_base', type=str)
    parser.add_argument('--target_set', default='noisy_novel', type=str)

    parser.add_argument('--noisy_frac', default=0, type=float)

    parser.add_argument('--similarity_pretrained', type=str, default='../saves/naive_NoisyNovel_CUB_lr0.0001_b64_wd0.0001_10280027/naive_NoisyNovel_CUB_lr0.0001_b64_wd0.0001_10280027_best.pth')
    return parser

def main():
    parser = get_base_parser()
    parser = add_base_train(parser)
    parser = add_similarity_train(parser)
    args = parser.parse_args()

    dataset_str = "Air"

    args.beta = 0.
    args.lr = 5e-3
    args.domain_lr = 5e-3
    args.batch_size = 50
    args.lr_interval = 25
    args.num_epoch = 300
    args.head_type = 4
    args.batch_class_num_B = 2
    args.similarity_pretrained = "../saves/naive_NoisyNovel_Air_lr0.005_b192_wd0.0001_12230007/naive_NoisyNovel_Air_lr0.005_b192_wd0.0001_12230007_0.8266.pth"
    # args.similarity_pretrained = "../saves/naive_NoisyNovel_Car_lr0.005_b192_wd0.0001_12230005/naive_NoisyNovel_Car_lr0.005_b192_wd0.0001_12230005_0.7989.pth"
    args.data_path = f'/home/zthang/zthang/SimTrans-Weak-Shot-Classification/workspace/dataset/{dataset_str}'

    init_log(args.exp_name)
    log(args)
    occupy(args.occupy)   # ????????????????????????
    set_seeds(args.seed)

    data_helper = get_data_helper(args)
    log(data_helper)

    # base_train_loader = data_helper.get_clean_base_loader()   # base training set??????150??? ??????30?????????
    #
    # base_test_loader = data_helper.get_base_test_loader()     # base test set 150??? 4326?????????
    #
    simnet = GANSimilarityNet(args).cuda()
    # dir_name = f"../image_embeddings/{dataset_str}/ori_pretrained"
    # if not os.path.exists(dir_name):
    #     os.makedirs(dir_name)
    # save_image_embedding(simnet, dir_name, "base_train", base_train_loader)
    # save_image_embedding(simnet, dir_name, "base_test", base_test_loader)
    #
    # base_train_image, base_test_image, base_train_label, base_test_label = get_dataset(dir_name)
    #
    # base_train_image = base_train_image.to("cpu")
    # make_graph_file(f"{dataset_str}/base_train_graph_mean_ori_pretrained", base_train_image)
    # base_test_image = base_test_image.to("cpu")
    # make_graph_file(f"{dataset_str}/base_test_graph_mean_ori_pretrained", base_test_image)

    for category_name in data_helper.novel_categories:
        dir_name = f"../image_embeddings/{dataset_str}/ori_pretrained/{category_name}"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        category_loader = data_helper.get_noisy_novel_sample_loader(category_name)
        save_image_embedding(simnet, dir_name, "web", category_loader)
        image, label = get_novel_dataset(dir_name)
        web_image = image.to("cpu")
        make_graph_file(f"{dataset_str}/novel/{category_name}", web_image)



if __name__ == '__main__':
    main()
