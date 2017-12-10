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
import options
from model.lstm import LSTM
from model.attention_lstm import AttentionPoolingLSTM


def predict(model, data_loader, t_map, cuda=False):
    ivt_t_map = {v:k for k, v in t_map.items()}
    model.eval()
    y_true = []
    y_pred = []
    indices = []
    for sample in tqdm(chain.from_iterable(data_loader)):
        feature = autograd.Variable(sample['feature'])
        position = autograd.Variable(sample['position'])
        target = sample['target']
        idx = sample['index']
        if cuda:
            feature = feature.cuda()
            position = position.cuda()
        output, _ = model(feature, position)
        _, pred = torch.max(output.data, dim=1)
        if cuda:
            pred = pred.cpu()
        y_true.append(target.numpy().flatten().tolist())
        y_pred.append(pred.numpy().tolist())
        indices.append(idx.numpy().tolist())
    
    y_true = chain.from_iterable(y_true)
    y_pred = chain.from_iterable(y_pred)
    indices = list(chain.from_iterable(indices))
    y_true = [y for _, y in sorted(zip(indices, [ivt_t_map[i] for i in y_true]))]
    y_pred = [y for _, y in sorted(zip(indices, [ivt_t_map[i] for i in y_pred]))]
    
    return y_true, y_pred

def main():
    parser = options.get_parser('Generator')
    options.add_dataset_args(parser)
    options.add_preprocessing_args(parser)
    options.add_model_args(parser)
    options.add_checkpoint_args(parser)
    options.add_generation_args(parser)
    
    args = parser.parse_args()
    print(args)
    
    args.cuda = not args.disable_cuda and torch.cuda.is_available()
    
    
    caseless = args.caseless
    batch_size = args.batch_size
    
    
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
    feature_map = checkpoint_file['f_map']
    target_map = checkpoint_file['t_map']
    test_features, test_targets = utils.build_corpus(test_corpus, feature_map, target_map, caseless)
    
    # train/val split
    test_loader = utils.construct_bucket_dataloader(test_features, test_targets, feature_map['PAD'], batch_size, args.position_bound, is_train=False)
    
    # build model
    vocab_size = len(feature_map)
    tagset_size = len(target_map)
    model = utils.build_model(args, vocab_size, tagset_size)
    
    
    # load states
    model.load_state_dict(checkpoint_file['state_dict'])
    
    if args.cuda:
        model.cuda()
    
    y_true, y_pred = predict(model, test_loader, target_map, cuda=args.cuda)
    assert len(y_pred) == len(test_corpus), 'length of prediction is inconsistent with that of data set'
    # write result: sent_id|e1|e2|ddi|type
    with open(args.analysis_file, 'w') as f:
        f.write(' | '.join(['sent_id', 'e1', 'e2', 'target', 'pred']))
        f.write('\n')
        for tup, target, pred in zip(test_raw_corpus, y_true, y_pred):
            if target != pred:
                f.write(' '.join(tup.sent))
                f.write('\n')
                f.write(' | '.join([tup.sent_id, tup.e1, tup.e2, target, pred]))
                f.write('\n\n')

if __name__ == '__main__':
    main()