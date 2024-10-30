# 训练器

import gc
import logging
import math
import os
import sys
import gc       # 垃圾收集器
import time
import random
import numpy
from sklearn import metrics
from time import strftime, localtime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import BertModel

from data_utils import Tokenizer4Bert, ABSA_Chinese_Dataset

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


class Trainer:
    def __init__(self, opt):
        self.opt = opt
        tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
        bert = BertModel.from_pretrained(opt.pretrained_bert_name)
        self.model = opt.model_class(bert, opt).to(opt.device)
        self.train_set = ABSA_Chinese_Dataset(opt.dataset_file['train'], tokenizer, is_chinese=opt.is_chinese)
        self.test_set = ABSA_Chinese_Dataset(opt.dataset_file['test'], tokenizer, is_chinese=opt.is_chinese)
        assert 0 <= opt.valid_ratio < 1
        if opt.valid_ratio > 0:
            valid_set_len = int(len(self.train_set) * opt.valid_ratio)
            self.train_set, self.valid_set = random_split(self.train_set, (len(self.train_set)-valid_set_len, valid_set_len))
        else:
            self.valid_set = self.test_set

        if opt.device.type == 'cuda':
            logger.info('>> cuda memory allocated: {}(Byte)'.format(torch.cuda.memory_allocated(device=opt.device.index)))
        self._print_args()

    def _print_args(self):
        n_trainable_parameters, n_nontrainable_parameters = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_parameters += n_params
            else:
                n_nontrainable_parameters += n_params
        logger.info('>> n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_parameters, n_nontrainable_parameters))
        logger.info('>> training arguments:')
        for arg in vars(self.opt):
            logger.info('>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for child in self.model.children():
            if type(child) != BertModel:
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.opt.initializer(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _train(self, criterion, optimizer, train_data_loader, val_data_loader):
        max_val_acc, max_val_f1 = 0, 0
        max_val_epoch, global_step = 0, 0
        path = None
        for i_epoch in range(self.opt.num_epoch):
            logger.info('>' * 100)
            logger.info('>> epoch: {}'.format(i_epoch))
            n_correct, n_total, loss_total = 0, 0, 0
            self.model.train()
            for i_batch, batch in enumerate(train_data_loader):
                global_step += 1
                optimizer.zero_grad()
                inputs = [batch[col].to(self.opt.device) for col in self.opt.inputs_format]
                outputs = self.model(inputs)
                targets = batch['polarity'].to(self.opt.device)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                n_correct = n_correct + (torch.argmax(outputs, -1) == targets).sum().item()
                n_total += len(outputs)
                loss_total += loss.item() * len(outputs)

                if global_step % self.opt.log_step == 0:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    logger.info('loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc))

                del outputs, loss, inputs
                torch.cuda.empty_cache()
                gc.collect()

            val_acc, val_f1 = self._evaluate_acc_f1(val_data_loader)
            logger.info('>> val_acc: {:.4f}, val_f1: {:.4f}'.format(val_acc, val_f1))
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                max_val_epoch = i_epoch
                if not os.path.exists('state_dict_param'):
                    os.mkdir('state_dict_param')

                # acc超过0.8保存模型
                # if max_val_acc >= 0.97:
                path = 'state_dict_param/{0}_{1}_{2}_val_acc_{3}_f1_{4}'.format(self.opt.model_name, self.opt.dataset,
                                                                           self.opt.threshold, round(val_acc, 4),
                                                                           round(val_f1, 4))
                torch.save(self.model.state_dict(), path)
                logger.info('>> saved: {}'.format(path))

                if val_f1 > max_val_f1:
                    max_val_f1 = val_f1
            logger.info('>> best_val_acc: {:.4f}, val_f1: {:.4f}'.format(max_val_acc, max_val_f1))
            if i_epoch - max_val_epoch >= self.opt.patience:
                print('>> early stop!')
                break
        return path


    def _evaluate_acc_f1(self, data_loader, _istest=False):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        self.model.eval()
        with torch.no_grad():
            for i_batch, t_batch in enumerate(data_loader):
                t_inputs = [t_batch[col].to(self.opt.device) for col in self.opt.inputs_format]
                t_targets = t_batch['polarity'].to(self.opt.device)
                t_outputs = self.model(t_inputs)
                n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)
            acc = n_correct / n_total
            f1 = metrics.f1_score(t_targets_all.cpu(),
                                  torch.argmax(t_outputs_all, -1).cpu(),
                                  labels=list(i for i in range(self.opt.n_class)),
                                  average='macro')
            return acc, f1

    def run(self):
        criterion = nn.CrossEntropyLoss()
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.lr, weight_decay=self.opt.l2reg)
        train_data_loader = DataLoader(dataset=self.train_set, batch_size=self.opt.batch_size, shuffle=True)
        test_data_loader = DataLoader(dataset=self.test_set, batch_size=self.opt.batch_size, shuffle=False)
        valid_data_loader = DataLoader(dataset=self.valid_set, batch_size=self.opt.batch_size, shuffle=False)
        # self._reset_params()
        best_model_path = self._train(criterion, optimizer, train_data_loader, valid_data_loader)

        # if best_model_path is not None:
        self.model.load_state_dict(torch.load(best_model_path))
        start_time = time.time()
        test_acc, test_f1 = self._evaluate_acc_f1(test_data_loader, True)
        cost_time = time.time() - start_time
        logger.info('>> test_acc: {:.4f}, test_f1: {:.4f}, test_time: {:.4f}'.format(test_acc, test_f1, cost_time))
        # return test_acc, test_f1
        # else:
        #     return 0, 0


