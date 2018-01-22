#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build argument parser
"""

import argparse

import criterion
from trainer import BasicTrainer

def get_parser(desc):
    parser = argparse.ArgumentParser(description='Relation Extraction Toolkit -- ' + desc)
    parser.add_argument('--disable-cuda', action='store_true', 
                        help='Do not use gpu resources')
    
    return parser

def add_dataset_args(parser):
    group = parser.add_argument_group('Dataset and data loading')
    group.add_argument('--raw_dir', default='../../data/re/drugddi2013/raw', 
                       help='raw corpus directory')
    group.add_argument('--processed_dir', default='../../data/re/drugddi2013/processed', 
                       help='processed data directory')
    group.add_argument('--emb_file', default='', help='path to load pretrained word embedding')
    group.add_argument('--batch_size', type=int, default=128, 
                       help='batch size')
    
    return group

def add_preprocessing_args(parser):
    group = parser.add_argument_group('Preprocessing')
    group.add_argument('--train_size', type=float, default=0.8, 
                       help='split train corpus into train/val set according to the ratio')
    group.add_argument('--caseless', action='store_true', 
                       help='caseless or not')
    group.add_argument('--position', action='store_true', 
                       help='use position feature')
    group.add_argument('--min_count', type=int, default=3, 
                       help='exclude words with frequency less than min count')
    group.add_argument('--const', action='store_true',
                       help='parse with constituency parser')
    group.add_argument('--dep', action='store_true',
                       help='parse with dependency parser')
    
    return group

def add_optimization_args(parser):
    group = parser.add_argument_group('Optimization')
    group.add_argument('--optimizer', default='nag', choices=BasicTrainer.OPTIMIZERS,
                       help='optimizer {}'.format(', '.join(BasicTrainer.OPTIMIZERS)))
    group.add_argument('--lr_scheduler', default='lambdalr', choices=BasicTrainer.LR_SCHEDULER,
                       help='lr scheduler {}'.format(', '.join(BasicTrainer.LR_SCHEDULER)))
    group.add_argument('--lr', type=float, default=0.25, 
                       help='learning rate')
    group.add_argument('--lr_decay', type=float, default=0.5, 
                       help='decay ratio of learning rate')
    group.add_argument('--momentum', type=float, default=0.9, 
                       help='momentum for sgd')
    group.add_argument('--clip_grad_norm', type=float, default=0.5, 
                       help='clip gradient norm')
    group.add_argument('--num_epoch', type=int, default=200, 
                       help='number of epochs')
    group.add_argument('--patience', type=int, default=15, 
                       help='patience for early stop')
    group.add_argument('--disable_fine_tune', action='store_true', 
                       help='Disable fine tuning word embedding')
    group.add_argument('--weight_decay', type=float, default=0.001, 
                       help='l2 regularization')
    group.add_argument('--adam_betas', default='(0.9, 0.999)',
                       help='coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))')
    group.add_argument('--class_weight', action='store_true', 
                       help='specify class weight in loss')
    group.add_argument('--loss', default='crossentropy', choices=criterion.CRITERION,
                       help='loss {}'.format(', '.join(criterion.CRITERION)))
    group.add_argument('--margin', type=float, default=1,
                       help='margin for margin loss')
    
    return group

def add_model_args(parser):
    group = parser.add_argument_group('Model')
    group.add_argument('--model', default='InterAttentionLSTM', choices=['lstm', 'InterAttentionLSTM', 'AttentionPoolingLSTM'],
                       help='specify model')
    group.add_argument('--embedding_dim', type=int, default=100, 
                       help='embedding dimension')
    group.add_argument('--relation_dim', type=int, default=100,
                       help='relation embedding dimension')
    group.add_argument('--position_dim', type=int, default=20, 
                       help='position embedding dimension')
    group.add_argument('--position_bound', type=int, default=200, 
                       help='relative position in [-200, 200]; if out of range, cast to min/max')
    group.add_argument('--hidden_dim', type=int, default=100, 
                       help='hidden layer dimension')
    group.add_argument('--rnn_layers', type=int, default=1, 
                       help='number of rnn layers')
    group.add_argument('--dropout_ratio', type=float, default=0.4, 
                       help='dropout ratio')
    group.add_argument('--rand_embedding', action='store_true', 
                       help='use randomly initialized word embeddings')
    group.add_argument('--att_hidden_dim', type=int, default=200, 
                       help='attention hidden dimension')
    group.add_argument('--num_hops', type=int, default=1, 
                       help='number of hops of attention')
    group.add_argument('--diagonal', action='store_true',
                       help='use diagonal bilinear for inter attention')
    group.add_argument('--sent_repr', type=str, default='max', choices=['max', 'concat'],
                       help='specify way to represent sentence')
    
    return group

def add_checkpoint_args(parser):
    group = parser.add_argument_group('Checkpointing')
    group.add_argument('--checkpoint', default='./checkpoint/re', 
                       help='path to checkpoint prefix')
    group.add_argument('--load_checkpoint', default='', 
                       help='path to load checkpoint')
    
    return group

def add_generation_args(parser):
    group = parser.add_argument_group('Generation')
    group.add_argument('--predict_file', default='../../data/drugddi2013/re/task9.2_GROUP_1.txt', 
                       help='path to output predicted result')
    group.add_argument('--att_file', default='../../data/drugddi2013/re/att_scores.txt',
                       help='path to attention scores file')
    group.add_argument('--error_file', default='../../data/drugddi2013/re/error.txt', 
                       help='path to analysis')
    
    return group
        

