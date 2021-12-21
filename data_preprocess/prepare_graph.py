import pickle
import sys
import warnings
import os

sys.path.append(".")
warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

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

    args.beta = 0.
    args.lr = 5e-3
    args.domain_lr = 5e-3
    args.batch_size = 50
    args.lr_interval = 25
    args.num_epoch = 300
    args.head_type = 4
    args.batch_class_num_B = 2
    args.similarity_pretrained = "../saves/pretrained/CUB/pretrained_84.5.pth"

    init_log(args.exp_name)
    log(args)
    occupy(args.occupy)   # 判断显存是否充足
    set_seeds(args.seed)

    data_helper = get_data_helper(args)
    log(data_helper)

    # base_train_loader = data_helper.get_clean_base_loader()   # base training set共计150类 每类30张图片
    # novel_train_loader = data_helper.get_noisy_novel_loader()  # novel web images 共计50类 每类1000张图片
    #
    # base_test_loader = data_helper.get_base_test_loader()     # base test set 150类 4326张图片
    # novel_test_loader = data_helper.get_novel_test_loader()   # novel test set 50类 1468张图片
    #
    # simnet = GANSimilarityNet(args).cuda()
    #
    for category_name in data_helper.novel_categories:
        dir_name = f"image_embeddings/CUB/ori_pretrained/{category_name}"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        category_loader = data_helper.get_nois
    # save_image_embedding(simnet, dir_name, "base_train", base_train_loader)
    # save_image_embedding(simnet, dir_name, "base_test", base_test_loader)


    # base_train_image, base_test_image, novel_train_image, novel_test_image, \
    # base_train_label, base_test_label, novel_train_label, novel_test_label = get_dataset("image_embeddings/CUB/")
    base_train_image, base_test_image, base_train_label, base_test_label = get_dataset(dir_name)


    base_train_image = base_train_image.to("cpu")
    make_graph_file("CUB/base_train_graph_mean_ori_pretrained", base_train_image)
    base_test_image = base_test_image.to("cpu")
    make_graph_file("CUB/base_test_graph_mean_ori_pretrained", base_test_image)
    # novel_train_image = novel_train_image.to("cpu")
    # make_graph_file("CUB/novel_train_graph_mean", novel_train_image)
    # novel_test_image = novel_test_image.to("cpu")
    # make_graph_file("CUB/novel_test_graph_mean", novel_test_image)


if __name__ == '__main__':
    main()
