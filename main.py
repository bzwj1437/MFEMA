#!/usr/bin/env/ python3
# _*_ coding: utf-8 _*_
# @Author: bzwj
# @Time: 2024/1/23 17:07

import os
import sys
import random
import argparse
import logging

import numpy
import torch

from time import strftime, localtime
from models import ABC_BERT, CBA_BERT, ABC_NO_A, ABC_NO_B, AOAN
from trainer import Trainer

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

os.environ["CUDA_DEVICES_ORDER"] = "PCI_BUS_IS"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def parameters():
    parser = argparse.ArgumentParser(description='参数及说明')
    parser.add_argument('--model_name', default='AOAN_bert', type=str)
    parser.add_argument('--dataset', default='camera', type=str, help='camera, car, notebook, phone')
    parser.add_argument('--optimizer', default='adam', type=str, help='优化器')
    parser.add_argument('--initializer', default='xavier_uniform_', type=str, help='参数初始化方式')
    parser.add_argument('--lr', default=4e-05, type=float, help='学习率')
    parser.add_argument('--l2reg', default=1e-05, type=float, help='L2正则化')
    parser.add_argument('--dropout', default=0, type=float, help='丢弃率')
    parser.add_argument('--num_epoch', default=30, type=int, help='对于non-BERT模型，可尝试更大的数值')
    parser.add_argument('--batch_size', default=16, type=int, help='批量大小')
    parser.add_argument('--log_step', default=10, type=int)
    parser.add_argument('--bert_dim', default=768, type=int, help='BERT预训练词向量维度, large:1024, base:768')
    parser.add_argument('--pretrained_bert_name', default='./hfl/chinese-roberta-wwm-ext', type=str)
    parser.add_argument('--max_seq_len', default=50, type=int, help='句子填充的最大长度')
    parser.add_argument('--n_class', default=2, type=int, help='情感极性的类别数')
    parser.add_argument('--patience', default=5, type=int, help='跟early stopping相关的忍耐度')
    parser.add_argument('--device', default='cuda:0', type=str, help='使用GPU训练')
    parser.add_argument('--seed', default=2024, type=int, help='随机种子值，用于固定随机值')
    parser.add_argument('--valid_ratio', default=0, type=float, help='取值0-1之间')
    parser.add_argument('--is_chinese', default=True, type=bool, help='Chinese or English')
    # ABC模型超参数
    parser.add_argument('--threshold', default=10, type=int)
    parser.add_argument('--n_layers', default=1, type=int)
    parser.add_argument('--d_model', default=768, type=int)
    parser.add_argument('--filter_num', default=256, type=int)
    opt = parser.parse_args([])

    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(opt.seed)

    model_classes = {
        'ABC_bert': ABC_BERT,
        'CBA_bert': CBA_BERT,
        'ABC_NO_A_bert': ABC_NO_A,
        'ABC_NO_B_bert': ABC_NO_B,
        'AOAN_bert': AOAN,
    }

    dataset_files = {
        'twitter': {
            'train': './datasets/acl-14-short-data/train.raw',  # max_seq_len: 84
            'test': './datasets/acl-14-short-data/test.raw'     # max_seq_len: 53
        },
        'restaurant': {
            'train': './datasets/semeval14/Restaurants_Train.xml.seg',      # max_seq_len: 99
            'test': './datasets/semeval14/Restaurants_Test_Gold.xml.seg'    # max_seq_len: 82
        },
        'laptop': {
            'train': './datasets/semeval14/Laptops_Train.xml.seg',          # max_seq_len: 92
            'test': './datasets/semeval14/Laptops_Test_Gold.xml.seg'        # max_seq_len: 100
        },
        'camera': {
            'train': './datasets/Chinese/camera/camera.train.txt',
            'test': './datasets/Chinese/camera/camera.test.txt'
        },
        'car': {
            'train': './datasets/Chinese/car/car.train.txt',
            'test': './datasets/Chinese/car/car.test.txt'
        },
        'notebook': {
            'train': './datasets/Chinese/notebook/notebook.train.txt',
            'test': './datasets/Chinese/notebook/notebook.test.txt'
        },
        'phone': {
            'train': './datasets/Chinese/phone/phone.train.txt',
            'test': './datasets/Chinese/phone/phone.test.txt'
        }
    }

    inputs_formats = {
        'ABC_bert': ['concat_bert_indices', 'concat_segments_indices', 'text_bert_indices', 'aspect_bert_indices'],
        'CBA_bert': ['concat_bert_indices', 'concat_segments_indices', 'text_bert_indices', 'aspect_bert_indices'],
        'ABC_NO_A_bert': ['concat_bert_indices', 'concat_segments_indices', 'text_bert_indices', 'aspect_bert_indices'],
        'ABC_NO_B_bert': ['concat_bert_indices', 'concat_segments_indices', 'text_bert_indices', 'aspect_bert_indices'],
        'AOAN_bert': ['concat_bert_indices', 'concat_segments_indices', 'text_bert_indices', 'aspect_bert_indices']
    }

    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }

    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamw': torch.optim.AdamW,
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }

    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_format = inputs_formats[opt.model_name]      # 模型输入数据的格式
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    return opt


if __name__ == '__main__':
    opt = parameters()

    # # 探讨threshold超参数影响
    # for i in [128, 256, 512, 768]:
    #     opt.filter_num = i
    #
    #     log_file = './log_param/{}-{}-{}.log'.format(opt.model_name, opt.dataset, strftime("%y%m%d-%H%M", localtime()))
    #     logger.addHandler(logging.FileHandler(log_file))
    #
    #     trainer = Trainer(opt)
    #     trainer.run()

    log_file = './log_param/{}-{}-{}.log'.format(opt.model_name, opt.dataset, strftime("%y%m%d-%H%M", localtime()))
    logger.addHandler(logging.FileHandler(log_file))

    trainer = Trainer(opt)
    trainer.run()


