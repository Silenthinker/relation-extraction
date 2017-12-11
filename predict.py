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
from trainer import Trainer
from model.lstm import LSTM
from model.attention_lstm import AttentionPoolingLSTM

def predict(trainer, data_loader, t_map, cuda=False):
    ivt_t_map = {v:k for k, v in t_map.items()}
    y_true = []
    y_pred = []
    att_weights = []
    indices = []
    for sample in tqdm(chain.from_iterable(data_loader)):
        target = sample['target']
        idx = sample['index']
        output_dict, pred = trainer.pred_step(sample)
        att_weight = output_dict['att_weight'].data
        if cuda:
            pred = pred.cpu()
            att_weight = att_weight.cpu()
        y_true.append(target.numpy().flatten().tolist())
        y_pred.append(pred.numpy().tolist())
        indices.append(idx.numpy().tolist())
        att_weights.append(att_weight.numpy().tolist())
    
    y_true = chain.from_iterable(y_true)
    y_pred = chain.from_iterable(y_pred)
    indices = list(chain.from_iterable(indices))
    att_weights = chain.from_iterable(att_weights)
    y_true = [y for _, y in sorted(zip(indices, [ivt_t_map[i] for i in y_true]))]
    y_pred = [y for _, y in sorted(zip(indices, [ivt_t_map[i] for i in y_pred]))]
    att_weights = [y for _, y in sorted(zip(indices, att_weights))]
    
    return y_true, y_pred, att_weights

def main():
    parser = options.get_parser('Generator')
    options.add_dataset_args(parser)
    options.add_preprocessing_args(parser)
    options.add_model_args(parser)
    options.add_optimization_args(parser)
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
    # loss
    criterion = utils.build_loss(args)
    
    # load states
    model.load_state_dict(checkpoint_file['state_dict'])
    
    # trainer
    trainer = Trainer(args, model, criterion)
    
    if args.cuda:
        model.cuda()
    
    y_true, y_pred, att_weights = predict(trainer, test_loader, target_map, cuda=args.cuda)
    assert len(y_pred) == len(test_corpus), 'length of prediction is inconsistent with that of data set'
    # prediction
    print('Predicting...')
    assert len(y_pred) == len(test_corpus), 'length of prediction is inconsistent with that of data set'
    # write result: sent_id|e1|e2|ddi|type
    with open(args.predict_file, 'w') as f:
        for tup, pred in zip(test_raw_corpus, y_pred):
            ddi = 0 if pred == 'null' else 1
            f.write('|'.join([tup.sent_id, tup.e1, tup.e2, str(ddi), pred]))
            f.write('\n')

    # error analysis
    print('Analyzing...')
    with open(args.error_file, 'w') as f:
        f.write(' | '.join(['sent_id', 'e1', 'e2', 'target', 'pred']))
        f.write('\n')
        for tup, target, pred, att_weight in zip(test_raw_corpus, y_true, y_pred, att_weights):
            if target != pred:
                size = len(tup.sent)
                f.write('{}\n'.format(' '.join(tup.sent)))
                f.write('{}\n'.format(' '.join(map(lambda x: str(round(x, 4)), att_weight[:size]))))
                f.write('{}\n\n'.format(' | '.join([tup.sent_id, tup.e1, tup.e2, target, pred])))
            
    # attention
    print('Writing attention scores...')
    with open(args.att_file, 'w') as f:
        f.write(' | '.join(['target', 'sent', 'att_weight']))
        f.write('\n')
        for tup, target, pred, att_weight in zip(test_raw_corpus, y_true, y_pred, att_weights):
            if target == pred and target != 'null':
                size = len(tup.sent)
                f.write('{}\n'.format(target))
                f.write('{}\n'.format(' '.join(tup.sent)))
                f.write('{}\n\n'.format(' '.join(map(lambda x: str(round(x, 4)), att_weight[:size]))))

if __name__ == '__main__':
    main()