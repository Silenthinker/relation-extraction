#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jyao
"""

from __future__ import print_function

import os
import time
from tqdm import tqdm
import collections

from itertools import chain

import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

import utils
import options
from model.tree_lstm import RelationTreeLSTM
import data.ddi2013 as ddi2013
from trainer import TreeTrainer

def main():
    parser = options.get_parser('Trainer')
    options.add_dataset_args(parser)
    options.add_preprocessing_args(parser)
    options.add_model_args(parser)
    options.add_optimization_args(parser)
    options.add_checkpoint_args(parser)
    
    args = parser.parse_args()
    print(args)
    
    args.cuda = not args.disable_cuda and torch.cuda.is_available()
    
    if args.cuda:
        torch.backends.cudnn.benchmark = True
        
    # checkpoint
    checkpoint_dir = os.path.dirname(args.checkpoint)
    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    
    # load dataset
    train_raw_corpus, val_raw_corpus, test_raw_corpus = utils.load_corpus(args.processed_dir)
    assert train_raw_corpus and val_raw_corpus and test_raw_corpus, 'Corpus not found, please run preprocess.py to obtain corpus!'
    train_corpus = [(line.sent, line.type, line.p1, line.p2) for line in train_raw_corpus]
    val_corpus = [(line.sent, line.type, line.p1, line.p2) for line in val_raw_corpus]    
    
    start_epoch = 0
    caseless = args.caseless
    batch_size = args.batch_size
    num_epoch = args.num_epoch
    
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
            split_path = os.path.join(data_dir, split + '.pth')
            split_dir = os.path.join(data_dir, split)
            if os.path.isfile(split_path):
                print('Found saved dataset, loading from {}'.format(split_path))
                dataset = torch.load(split_path)
            else:
                print('Building dataset from scratch...')
                dataset = ddi2013.DDI2013TreeDataset(split_dir, feature_map, args.caseless)
                print('Save dataset to {}'.format(split_path))
                torch.save(dataset, split_path)
            if dataloader:
                return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=split != 'test', collate_fn=dataset.collate, drop_last=False)
            else:
                return dataset
        return load_dataset('train'), load_dataset('val'), load_dataset('test')
        
    
    train_loader, val_loader, test_loader = load_datasets(args.processed_dir, dataloader=True)            
    
    # build model
    vocab_size = len(feature_map)
    tagset_size = len(target_map)
    model = RelationTreeLSTM(vocab_size, tagset_size, args)
    
    # loss
    criterion = utils.build_loss(args, class_weights=class_weights)
    
    # load states
    if os.path.isfile(args.load_checkpoint):
        print('Loading checkpoint file from {}...'.format(args.load_checkpoint))
        checkpoint_file = torch.load(args.load_checkpoint)
        start_epoch = checkpoint_file['epoch'] + 1
        model.load_state_dict(checkpoint_file['state_dict'])
    #    optimizer.load_state_dict(checkpoint_file['optimizer'])
    else:
        print('no checkpoint file found: {}, train from scratch...'.format(args.load_checkpoint))
        if not args.rand_embedding:
            pretrained_word_embedding, in_doc_word_indices = utils.load_word_embedding(args.emb_file, feature_map, args.embedding_dim)
            print(pretrained_word_embedding.size())
            print(vocab_size)
            model.load_pretrained_embedding(pretrained_word_embedding)
            if args.disable_fine_tune:
                model.update_part_embedding(in_doc_word_indices) # update only non-pretrained words
        model.rand_init(init_embedding=args.rand_embedding)
    
    for i in test_loader:
        print(i)
        break
    '''
    # trainer
    trainer = TreeTrainer(args, model, criterion)
    
    if os.path.isfile(args.load_checkpoint):
        dev_prec, dev_rec, dev_f1, _ = evaluate(trainer, val_loader, target_map, cuda=args.cuda)
        test_prec, test_rec, test_f1, _ = evaluate(trainer, test_loader, target_map, cuda=args.cuda)
        print('checkpoint dev_prec: {:.4f}, dev_rec: {:.4f}, dev_f1: {:.4f}, test_prec: {:.4f}, test_rec: {:.4f}, test_f1: {:.4f}'.format(
            dev_prec, dev_rec, dev_f1, test_prec, test_rec, test_f1))
    
    track_list = []
    best_f1 = float('-inf')
    patience_count = 0
    start_time = time.time()
    
    
    for epoch in range(start_epoch, num_epoch):
        epoch_loss = train(train_loader, trainer, epoch)
    
        # update lr
        trainer.lr_step()
           
        dev_prec, dev_rec, dev_f1, dev_loss = evaluate(trainer, val_loader, target_map, cuda=args.cuda)
        if dev_f1 >= best_f1:
            patience_count = 0
            best_f1 = dev_f1
    
            test_prec, test_rec, test_f1, _ = evaluate(trainer, test_loader, target_map, cuda=args.cuda)
    
            track_list.append({'epoch': epoch, 'loss': epoch_loss, 
                'dev_prec': dev_prec, 'dev_rec': dev_rec, 'dev_f1': dev_f1, 'dev_loss': dev_loss, 
                'test_prec': test_prec, 'test_rec': test_rec, 'test_f1': test_f1})
            print('epoch: {}, loss: {:.4f}, dev_f1: {:.4f}, dev_loss: {:.4f}, test_f1: {:.4f}\tsaving...'.format(epoch, epoch_loss, dev_f1, dev_loss, test_f1))
    
            try:
                utils.save_checkpoint({
                            'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'optimizer': trainer.optimizer.state_dict(),
                            'f_map': feature_map,
                            't_map': target_map,
                        }, {'track_list': track_list,
                            'args': vars(args)
                            }, args.checkpoint + '_lstm')
            except Exception as inst:
                print(inst)
        else:
            patience_count += 1
            track_list.append({'epoch': epoch,'loss': epoch_loss, 'dev_prec': dev_prec, 'dev_rec': dev_rec, 'dev_f1': dev_f1, 'dev_loss': dev_loss})
            print('epoch: {}, loss: {:.4f}, dev_f1: {:.4f}, dev_loss: {:.4f}'.format(epoch, epoch_loss, dev_f1, dev_loss))
    
        print('epoch: {} in {} take: {} s'.format(epoch, args.num_epoch, time.time() - start_time))
        if patience_count >= args.patience:
            break
    
    '''

if __name__ == '__main__':
    main()
## TODO: 
# tqdm updates loss, grad
# residual connection using rnncell
