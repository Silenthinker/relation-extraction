#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jyao
"""

from __future__ import print_function

import os
import sys
import time
import json
from tqdm import tqdm
import collections
from collections import namedtuple

from itertools import chain

import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from torch import multiprocessing as mp

import utils
import options
import meters
from model.tree_lstm import RelationTreeModel
import data.ddi2013 as ddi2013
from trainer import TreeTrainer

def predict(trainer, data_loader, t_map, cuda=False):
    y_true = []
    y_pred = []
    att_weights = []
    
    tot_length = len(data_loader)
    tot_loss = 0
    
    ivt_t_map = {v:k for k, v in t_map.items()}
    
    loss_meter = meters.AverageMeter()
    with tqdm(data_loader, total=tot_length) as pbar:
        for sample in pbar:
            target = sample['target']
            loss = trainer.valid_step(sample)
            output_dict, pred = trainer.pred_step(sample)
            trees = sample['tree']
            indices = [[t.idx for t in treelist] for treelist in trees]
            if 'att_weight' in output_dict:
                att_weight = [item.numpy().flatten().tolist() for item in output_dict['att_weight']]
                # order by indices
                att_weight = [[s for _, s in sorted(zip(i, a))] for i, a in zip(indices, att_weight)]
                if cuda:
                    att_weight = [item.cpu() for item in att_weight]
            
            if cuda:
                pred = pred.cpu() # cast back to cpu
                
            tot_loss += loss
            y_true.append(target.view(-1).numpy().tolist())
            y_pred.append(pred.view(-1).numpy().tolist())
            att_weights.append(att_weight)
            loss_meter.update(loss)
            pbar.set_postfix(collections.OrderedDict([
                    ('loss', '{:.4f} ({:.4f})'.format(loss, loss_meter.avg))
                    ]))     
    
    y_true = [ivt_t_map[i] for i in chain.from_iterable(y_true)]
    y_pred = [ivt_t_map[i] for i in chain.from_iterable(y_pred)]
    att_weights = list(chain.from_iterable(att_weights))
    
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
    
    model_path = args.load_checkpoint + '.model'
    args_path = args.load_checkpoint + '.json'
    with open(args_path, 'r') as f: 
        _args = json.load(f)['args']
    [setattr(args, k, v) for k,v in _args.items()]
    
    args.cuda = not args.disable_cuda and torch.cuda.is_available()
    
    if args.cuda:
        torch.backends.cudnn.benchmark = True
    
    # increase recursion depth
    sys.setrecursionlimit(10000)
    
    # load dataset
    train_raw_corpus, val_raw_corpus, test_raw_corpus = utils.load_corpus(args.processed_dir, ddi=False)
    assert train_raw_corpus and val_raw_corpus and test_raw_corpus, 'Corpus not found, please run preprocess.py to obtain corpus!'
    train_corpus = [(line.sent, line.type, line.p1, line.p2) for line in train_raw_corpus]
    val_corpus = [(line.sent, line.type, line.p1, line.p2) for line in val_raw_corpus]    
    
    caseless = args.caseless
    batch_size = args.batch_size
    
    # build vocab
    sents = [tup[0] for tup in train_corpus + val_corpus]
    feature_map = utils.build_vocab(sents, min_count=args.min_count, caseless=caseless)
    target_map = ddi2013.target_map
    
    # get class weights
    _, train_targets = utils.build_corpus(train_corpus, feature_map, target_map, caseless)
    class_weights = torch.Tensor(utils.get_class_weights(train_targets)) if args.class_weight else None
    
    # load dataset
    def load_datasets(data_dir, dataloader=True):
        """
        load train, val, and test dataset
        data_dir: dir of datasets
        dataloader: bool, True to return pytorch Dataloader
        """
        # splits = ['train', 'val', 'test']
        
        def load_dataset(split):
            _const = 'c' if not args.childsum_tree else ''
            split_path = os.path.join(data_dir, split + '.' + _const + 'pth')
            split_dir = os.path.join(data_dir, split)
            if os.path.isfile(split_path):
                print('Found saved dataset, loading from {}'.format(split_path))
                dataset = torch.load(split_path)
            else:
                print('Building dataset from scratch...')
                dataset = ddi2013.DDI2013TreeDataset(split_dir, feature_map, args.caseless, dep=args.childsum_tree)
                print('Save dataset to {}'.format(split_path))
                torch.save(dataset, split_path)
            if dataloader:
                return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=split != 'test', collate_fn=dataset.collate, drop_last=False)
            else:
                return dataset
        return load_dataset('train'), load_dataset('val'), load_dataset('test')
        
    
    _, _, test_loader = load_datasets(args.processed_dir, dataloader=True)            
    
    # build model
    vocab_size = len(feature_map)
    tagset_size = len(target_map)
    model = RelationTreeModel(vocab_size, tagset_size, args)
    
    # loss
    criterion = utils.build_loss(args, class_weights=class_weights)
    
    # load states
    assert os.path.isfile(model_path), "Checkpoint not found!"
    print('Loading checkpoint file from {}...'.format(model_path))
    checkpoint_file = torch.load(model_path)
    model.load_state_dict(checkpoint_file['state_dict'])
    
    # trainer
    trainer = TreeTrainer(args, model, criterion)
    
    # predict
    y_true, y_pred, att_weights = predict(trainer, test_loader, target_map, cuda=args.cuda)
    
    # prediction
    print('Predicting...')
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
                f.write('{}\n'.format(' '.join(tup.sent)))
                f.write('{}\n'.format(' | '.join([tup.sent_id, tup.e1, tup.e2, target, pred])))
                f.write('{}\n\n'.format(' '.join(map(lambda x: str(round(x, 4)), att_weight))))
    
    # attention
    print('Writing attention scores...')
    with open(args.att_file, 'w') as f:
        f.write(' | '.join(['target', 'sent', 'att_weight']))
        f.write('\n')
        for tup, target, pred, att_weight in zip(test_raw_corpus, y_true, y_pred, att_weights):
            if target == pred and target != 'null':
                f.write('{}\n'.format(target))
                f.write('{}\n'.format(' '.join(tup.sent)))
                f.write('{}\n'.format(' '.join(map(lambda x: str(round(x, 4)), att_weight))))
    

if __name__ == '__main__':
    main()
