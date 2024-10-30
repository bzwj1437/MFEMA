#数据处理模块

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


def pad_and_truncate(sequence, max_seq_len, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(max_seq_len)*value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-max_seq_len:]
    else:
        trunc = sequence[:max_seq_len]
    trunc = np.asarray(trunc, dtype=dtype)

    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x


class Tokenizer4Bert:
    def __init__(self, max_seq_len, pretrained_bert_name):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        self.max_seq_len = max_seq_len

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


class ABSA_Chinese_Dataset(Dataset):
    def __init__(self, fname, tokenizer, is_chinese=True):
        with open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
            lines = f.readlines()

        all_data = []
        for i in range(0, len(lines), 3):
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition('$T$')]
            aspect = lines[i+1].lower().strip()
            polarity = lines[i+2].strip()
            text = text_left + " " + aspect + " " + text_right

            text_indices = tokenizer.text_to_sequence(text)
            text_len = np.sum(text_indices != 0)
            aspect_indices = tokenizer.text_to_sequence(aspect)
            aspect_len = np.sum(aspect_indices != 0)

            text_bert_indices = tokenizer.text_to_sequence('[CLS] ' + text_left + ' ' + aspect + ' ' + text_right + ' [SEP]')
            concat_bert_indices = tokenizer.text_to_sequence('[CLS] ' + text_left + ' ' + aspect +' ' + text_right +' [SEP] ' + aspect + ' [SEP]')
            concat_segments_indices = [0] * (text_len + 2) + [1] * (aspect_len + 1)
            concat_segments_indices = pad_and_truncate(concat_segments_indices, tokenizer.max_seq_len)
            aspect_bert_indices = tokenizer.text_to_sequence('[CLS] ' + aspect + ' [SEP]')

            if is_chinese is True:
                polarity = int(polarity)
            else:
                polarity = int(polarity) + 1

            data = {
                'text_bert_indices': text_bert_indices,
                'concat_bert_indices': concat_bert_indices,
                'concat_segments_indices': concat_segments_indices,
                'aspect_bert_indices': aspect_bert_indices,
                'aspect_indices': aspect_indices,
                'polarity': polarity
            }
            all_data.append(data)
        self.data = all_data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

