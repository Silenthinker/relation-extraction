#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jyao
"""

from __future__ import print_function

import os
import time
from tqdm import tqdm

from itertools import chain

import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

import utils
from model.lstm import LSTM
from model.attention_lstm import AttentionLSTM

def predict(model, data_loader, t_map, cuda=False):
    ivt_t_map = {v:k for k, v in t_map.items()}
    model.eval()
    y_pred = []
    indices = []
    for (feature, position), _, idx in tqdm(chain.from_iterable(data_loader)):
        feature = autograd.Variable(feature)
        position = autograd.Variable(position)
        if cuda:
            feature = feature.cuda()
            position = position.cuda()
        output, _ = model(feature, position)
        _, pred = torch.max(output.data, dim=1)
        if cuda:
            pred = pred.cpu()
        y_pred.append(pred.numpy().tolist())
        indices.append(idx.numpy().tolist())
    y_pred = chain.from_iterable(y_pred)
    indices = chain.from_iterable(indices)
    return [y for _, y in sorted(zip(indices, [ivt_t_map[i] for i in y_pred]))]

parser = utils.build_parser()
args = parser.parse_args()
args.cuda = not args.disable_cuda and torch.cuda.is_available()

caseless = args.caseless
batch_size = args.batch_size
num_epoch = args.num_epoch
lr = args.lr
momentum = args.momentum
clip_grad_norm = args.clip_grad_norm

if os.path.isfile(args.load_checkpoint):
    print('Loading checkpoint file from {}...'.format(args.load_checkpoint))
    checkpoint_file = torch.load(args.load_checkpoint)
else:
    print('No checkpoint file found: {}'.format(args.load_checkpoint))
    raise OSError
    
_, test_raw_corpus = utils.load_corpus(args.train_path, args.test_path)
if not test_raw_corpus:
    test_raw_corpus = utils.preprocess_ddi(data_path=args.test_corpus_path, output_path=args.test_path, position=True)
test_corpus = [(line.sent, line.type, line.p1, line.p2) for line in test_raw_corpus]

# preprocessing
feature_mapping = checkpoint_file['f_map']
target_mapping = checkpoint_file['t_map']
test_features, test_targets = utils.build_corpus(test_corpus, feature_mapping, target_mapping, caseless)

# train/val split
test_loader = utils.construct_bucket_dataloader(test_features, test_targets, feature_mapping['PAD'], batch_size, args.position_bound, is_train=False)

# build model
vocab_size = len(feature_mapping)
tagset_size = len(target_mapping)
model = AttentionLSTM(vocab_size, tagset_size, args) if args.attention else LSTM(vocab_size, tagset_size, args)


# load states
model.load_state_dict(checkpoint_file['state_dict'])

if args.cuda:
    model.cuda()

y_pred = predict(model, test_loader, target_mapping, cuda=args.cuda)
assert len(y_pred) == len(test_corpus), 'length of prediction is inconsistent with that of data set'
# write result: sent_id|e1|e2|ddi|type
with open(args.predict_file, 'w') as f:
    for tup, pred in zip(test_raw_corpus, y_pred):
        ddi = 0 if pred == 'null' else 1
        f.write('|'.join([tup.sent_id, tup.e1, tup.e2, str(ddi), pred]))
        f.write('\n')

