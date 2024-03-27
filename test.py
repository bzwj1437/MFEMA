#!/usr/bin/env/ python3
# _*_ coding: utf-8 _*_
# @Author: bzwj
# @Time: 2024/1/29 21:21

import logging
import sys

import torch
import torch.nn as nn
from sklearn import metrics
from torch.utils.data import DataLoader
from main import parameters
from data_utils import Tokenizer4Bert, ABSA_Chinese_Dataset
from transformers import BertModel
from trainer import Trainer

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


if __name__ == '__main__':
    opt = parameters()
    tester = Trainer(opt)

    criterion = nn.CrossEntropyLoss()
    _params = filter(lambda p: p.requires_grad, tester.model.parameters())
    optimizer = tester.opt.optimizer(_params, lr=tester.opt.lr, weight_decay=tester.opt.l2reg)
    test_data_loader = DataLoader(dataset=tester.test_set,
                                  batch_size=tester.opt.batch_size,
                                  shuffle=False)
    best_model_path = './state_dicts/ABC_bert_camera_5_val_acc_0.9793_f1_0.9753'
    tester.model.load_state_dict(torch.load(best_model_path))
    test_acc, test_f1 = tester._evaluate_acc_f1(test_data_loader, True)
    logger.info('>> test_acc: {:.4f}, test_f1: {:.4f}'.format(test_acc, test_f1))
