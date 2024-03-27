#!/usr/bin/env/ python3
# _*_ coding: utf-8 _*_
# @Author: bzwj
# @Time: 2024/1/23 20:56

import numpy as np
import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertPooler, BertSelfAttention


class SelfAttention(nn.Module):
    def __init__(self, config, opt):
        super(SelfAttention, self).__init__()
        self.opt = opt
        self.config = config
        self.SA = BertSelfAttention(config)
        self.tanh = torch.nn.Tanh()

    def forward(self, inputs):
        zero_tensor = torch.tensor(np.zeros((inputs.size(0), 1, 1, self.opt.max_seq_len),
                                            dtype=np.float32), dtype=torch.float32).to(self.opt.device)
        SA_out = self.SA(inputs, zero_tensor)
        return self.tanh(SA_out[0])


class ABC_BERT(nn.Module):
    def __init__(self, bert, opt):
        super(ABC_BERT, self).__init__()
        self.bert_spc = bert
        self.opt = opt

        # 冻结bert模型的参数
        # for param in self.bert_spc.parameters():
        #     param.requires_grad = False

        self.dropout = nn.Dropout(opt.dropout)
        self.bert_SA = SelfAttention(bert.config, opt)
        self.linear_double = nn.Linear(opt.filter_num + opt.bert_dim*2, opt.d_model)
        # self.linear_single = nn.Linear(opt.d_model, opt.d_model)
        self.bert_pooler = BertPooler(bert.config)
        self.dense = nn.Linear(opt.d_model, opt.n_class)
        self.pool = nn.AvgPool1d(opt.threshold+1)

        self.convs = nn.ModuleList([nn.Conv1d(in_channels=opt.bert_dim,
                                              out_channels=opt.filter_num,
                                              kernel_size=(2*kernel+1,),
                                              padding=kernel).to(opt.device) for kernel in range(opt.threshold+1)])

    @staticmethod
    def conv_and_relu(x, conv):
        x = x.transpose(1, 2).contiguous()
        x = nn.ReLU()(conv(x))
        return x.transpose(1, 2).contiguous()

    def forward(self, inputs):
        concat_bert_indices = inputs[0]
        concat_segments_indices = inputs[1]
        text_local_indices = inputs[2]

        bert_spc_out, _ = self.bert_spc(concat_bert_indices,
                                        token_type_ids=concat_segments_indices,
                                        return_dict=False)
        bert_spc_out = self.dropout(bert_spc_out)
        local_span, _ = self.bert_spc(text_local_indices, return_dict=False)
        local_span = self.dropout(local_span)

        multi_feature = []
        for conv in self.convs:
            feature_vec = self.conv_and_relu(local_span, conv)
            enhanced_feature = torch.cat((feature_vec, local_span, bert_spc_out), dim=-1)
            mean_pool = self.linear_double(enhanced_feature)
            self_attention_out = self.bert_SA(mean_pool)
            pooled_out = self.bert_pooler(self_attention_out)
            dense_out = self.dense(pooled_out)
            multi_feature.append(dense_out)

        out = torch.stack(multi_feature, dim=-1)
        ensemble_out = self.pool(out)
        ensemble_out = ensemble_out.squeeze(-1)

        return ensemble_out




    # def shallow_feature_extraction(self, x, kernel):
    #     """
    #     多尺度特征提取
    #     :param x:       [batch_size, seq_len, d_model]
    #     :param kernel:  0-N
    #     :return:
    #     """
    #     x = x.transpose(1, 2).contiguous()      # [batch_size, d_model, seq_len]
    #     self.conv = nn.Conv1d(in_channels=self.opt.bert_dim,
    #                           out_channels=self.opt.filter_num,
    #                           kernel_size=kernel,
    #                           padding=(kernel-1)//2).to(self.opt.device)
    #     x = nn.ReLU()(self.conv(x))
    #     return x
    #
    # def forward(self, inputs):
    #     concat_bert_indices = inputs[0]
    #     concat_segments_indices = inputs[1]
    #     text_local_indices = inputs[2]
    #
    #     # with torch.no_grad():
    #     bert_spc_out, _ = self.bert_spc(concat_bert_indices, token_type_ids=concat_segments_indices, return_dict=False)
    #     bert_spc_out = self.dropout(bert_spc_out)
    #
    #     local_span, _ = self.bert_spc(text_local_indices, return_dict=False)
    #     local_span = self.dropout(local_span)
    #
    #     multi_feature = []
    #     for i_kernel in range(self.opt.threshold+1):
    #         feature_vec = self.shallow_feature_extraction(local_span, 2*i_kernel+1)
    #         feature_vec = feature_vec.transpose(1, 2).contiguous()
    #         enhanced_feature = torch.cat((feature_vec, local_span, bert_spc_out), dim=-1)    # 局部特征，原始词向量，全局词向量
    #         mean_pool = self.linear_double(enhanced_feature)
    #         self_attention_out = self.bert_SA(mean_pool)
    #         pooled_out = self.bert_pooler(self_attention_out)
    #         dense_out = self.dense(pooled_out)
    #         multi_feature.append(dense_out)
    #
    #     out = torch.stack(multi_feature, dim=-1)
    #     ensemble_out = self.pool(out)
    #     ensemble_out = ensemble_out.squeeze(-1)
    #
    #     return ensemble_out



