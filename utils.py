#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import math
from tqdm import tqdm
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F

from itertools import chain
from operator import itemgetter

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

from constants import CONSTANTS
import data.ddi2013 as ddi2013
from data.ddi2013 import DDI2013SeqDataset
from criterion import HingeLoss
from model.lstm import LSTM
from model.attention_lstm import InterAttentionLSTM, AttentionPoolingLSTM

''' Example
<document id="DrugDDI.d89" origId="Aciclovir">
    <sentence id="DrugDDI.d89.s0" origId="s0" text="Co-administration of probenecid with acyclovir has been shown to increase the mean half-life and the area under the concentration-time curve.">
        <entity id="DrugDDI.d89.s0.e0" origId="s0.p1" charOffset="21-31" type="drug" text="probenecid"/>
        <entity id="DrugDDI.d89.s0.e1" origId="s0.p2" charOffset="37-47" type="drug" text="acyclovir"/>
        <pair id="DrugDDI.d89.s0.p0" e1="DrugDDI.d89.s0.e0" e2="DrugDDI.d89.s0.e1" interaction="true"/>
    </sentence><sentence id="DrugDDI.d89.s1" origId="s1" text="Urinary excretion and renal clearance were correspondingly reduced."/>
    <sentence id="DrugDDI.d89.s2" origId="s2" text="The clinical effects of this combination have not been studied."/>
</document>
'''

## TODO: tokenization and remove punctuations which are not part of entity mentions

def make_dirs(dirs):
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
            
def build_vocab(sents, min_count=5, caseless=True):
    """
    Args:
        sents: list of list of strings, [[str]]
        min_count: threshold to replace rare words with UNK
    """
    counter = Counter()
    if caseless:
        counter.update([w.lower() for w in chain.from_iterable(sents)])
    else:
        counter.update(chain.from_iterable(sents))
    str2int = [w for w, c in counter.items() if c >= min_count]
    str2int = {w:i for i, w in enumerate(str2int, len(CONSTANTS))} # start from length of CONSTANTS
    str2int.update(CONSTANTS)
    
    return str2int
    
def build_features(raw_features, mapping, caseless=True):
    """
    Args:
        raw_features: list of list of strings
        mappling: dict mapping from word to index
    Return:
        [[int]]
    """
    def encode(w, mapping, caseless):
        if caseless:
            w = w.lower()
        return mapping.get(w, mapping['UNK'])
    
    return [list(map(lambda w:encode(w, mapping, caseless), sent)) for sent in raw_features]

# load dataset
def load_datasets(data_dir, train_size, args, feature_map, dataloader=True):
    """
    load train, val, and test dataset
    data_dir: dir of datasets
    train_size: float
    dataloader: bool, True to return pytorch Dataloader
    """
    # splits = ['train', 'val', 'test']
    
    def wrap_dataloader(dataset, shuffle):
        return torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, collate_fn=dataset.collate, drop_last=False)
    
    def load_dataset(split, dataloader):
        _const = 'c' if not args.childsum_tree else ''
        split_path = os.path.join(data_dir, split + '.' + _const + 'pth')
        split_dir = os.path.join(data_dir, split)
        if os.path.isfile(split_path):
            print('Found saved dataset, loading from {}'.format(split_path))
            dataset = torch.load(split_path)
        else:
            print('Building dataset from scratch...')
            dataset = ddi2013.DDI2013TreeDataset(feature_map, args.caseless, sp=args.sp, dep=args.childsum_tree, path=split_dir, )
            print('Save dataset to {}'.format(split_path))
            torch.save(dataset, split_path)
        if dataloader:
            return wrap_dataloader(dataset, shuffle=split != 'test')
        else:
            return dataset
    
    train, val, test = load_dataset('train', False), load_dataset('val', False), load_dataset('test', False)
    
    # concatenate and split
    sentences = train.sentences + val.sentences
    positions = train.positions + val.positions
    trees = train.trees + val.trees
    labels = torch.cat([train.labels, val.labels], dim=0)
    
    train_features, train_targets, val_features, val_targets = stratified_shuffle_split(list(zip(sentences, positions, trees)), labels.numpy().tolist(), train_size=train_size)
    train_targets, val_targets = torch.LongTensor(train_targets),  torch.LongTensor(val_targets)
    train_data, val_data = list(zip(*train_features)), list(zip(*val_features))
    train_data.append(train_targets)
    val_data.append(val_targets)
    
    train = ddi2013.DDI2013TreeDataset(feature_map, args.caseless, dep=args.childsum_tree, data=train_data)
    val = ddi2013.DDI2013TreeDataset(feature_map, args.caseless, dep=args.childsum_tree, data=val_data)
    
    return wrap_dataloader(train, True), wrap_dataloader(val, True), wrap_dataloader(test, False)
       
        
def map_iterable(iterable, mapping):
    """
    Args:
        iterable: iterable
        mapping: dict
    """
    return [mapping[k] for k in iterable]

def build_corpus(raw_corpus, feature_mapping, target_mapping, caseless):
    """
    build features and targets
    Args:
        raw_corpus: [([str], str)], list of tuple(features, target)
        or optionally with positions [int], [int]
    """
    raw_features = [tup[0] for tup in raw_corpus]
    raw_targets = [tup[1] for tup in raw_corpus]
    features = build_features(raw_features, feature_mapping, caseless)
    raw_positions = [chain.from_iterable([tup[2], tup[3]]) for tup in raw_corpus]
    features = [list(chain.from_iterable([f, p])) for f, p in zip(features, raw_positions)]
    targets = map_iterable(raw_targets, target_mapping)
    return features, targets    
    
def stratified_shuffle_split(features, targets, train_size=0.9):
    """
    Args:
        inputs: list
        targets: [int]
    Return:
        (train_features, train_targets), (val_features, val_targets)
    """
    def np2list(array):
        """
        numpy array to list
        """
        return array.tolist()

    X = np.arange(len(targets)) # serve as indices    
    y = np.array(targets)    
    train_index, val_index, _, _ = train_test_split(X, y, train_size=train_size, random_state=5)
    
    # split features
    train_features = []
    val_features = []
    for i in train_index:
        train_features.append(features[i])
    for i in val_index:
        val_features.append(features[i])
    
    #split targets
    train_targets = y[train_index]
    val_targets = y[val_index]
    return train_features, np2list(train_targets), val_features, np2list(val_targets)

def calc_threshold_mean(features):
    """
    calculate the threshold for bucket by mean
    """
    lines_len = list(map(lambda t: len(t)//3 + 1, features))
    average = int(sum(lines_len) / len(lines_len))
    lower_line = list(filter(lambda t: t < average, lines_len))
    upper_line = list(filter(lambda t: t > average, lines_len))
    if lower_line:
        lower_average = int(sum(lower_line) / len(lower_line))
    else:
        lower_average = None
    if upper_line:
        upper_average = int(sum(upper_line) / len(upper_line))
    else:
        upper_average = None
    max_len = max(lines_len)
    max_len = None if max_len == upper_average else max_len
    thresholds = [line_len for line_len in [lower_average, average, upper_average, max_len] if line_len]
    return sorted(list(set(thresholds)))

def construct_bucket_dataloader(input_features, input_targets, pad_feature, batch_size, position_bound, is_train=True):
    """
    Construct bucket
    input_features: [[int]], with concatenated word and position features
    """
    def pad_position(p_feature, threshold, cur_len, position_bound):
        """
        assign position to padded text with range check
        """
        p_feature = p_feature + list(range(p_feature[-1]+1, p_feature[-1]+1+threshold - cur_len))
        return list(map(lambda p: p if abs(p) <= position_bound else math.copysign(position_bound, p), p_feature))
        
    # encode and padding
    thresholds = calc_threshold_mean(input_features)
    buckets = [[[], [], [], [], []] for _ in range(len(thresholds))] # [[w_feature, target, p_feature, idx, mask]]
    for i, (feature, target) in enumerate(zip(input_features, input_targets)):
        assert len(feature) % 3 == 0, 'len(feature) % 3 != 0'
        cur_len = len(feature) // 3
        w_feature = feature[0:cur_len]
        p1_feature = feature[cur_len:-cur_len]
        p2_feature = feature[-cur_len:]
        idx = 0
        cur_len_1 = cur_len + 1
        while thresholds[idx] < cur_len_1:
            idx += 1
        buckets[idx][0].append(w_feature + [pad_feature] * (thresholds[idx] - cur_len))
        buckets[idx][1].append([target])
        buckets[idx][2].append(pad_position(p1_feature, thresholds[idx], cur_len, position_bound) +
               pad_position(p2_feature, thresholds[idx], cur_len, position_bound))
        buckets[idx][3].append(i)
        buckets[idx][4].append([1] * cur_len + [0] * (thresholds[idx] - cur_len))
    bucket_dataset = [DDI2013SeqDataset(torch.LongTensor(bucket[0]), torch.LongTensor(bucket[1]), torch.LongTensor(bucket[2]), bucket[3], torch.ByteTensor(bucket[4]))
                      for bucket in buckets]
    dataset_loader = [torch.utils.data.DataLoader(tup, batch_size, shuffle=is_train, drop_last=False) for tup in bucket_dataset]
    return dataset_loader

def load_word_embedding(file, feature_mapping, emb_dim, delimiter=' '):
    """
    load pretrained word embeddings from file
    """
    vocab_size = len(feature_mapping)
    word_embedding = torch.FloatTensor(vocab_size, emb_dim)
    init_embedding(word_embedding)
    in_doc_word_indices = []
    
    n_pretrain = 0
    with open(file, 'r') as f:
        for line in tqdm(f):
            line = line.strip().split(delimiter)
            w = line[0]
            if w in feature_mapping:
                n_pretrain += 1
                vec = list(map(lambda x:float(x), line[1:]))
                word_embedding[feature_mapping[w]] = torch.FloatTensor(vec)
                in_doc_word_indices.append(feature_mapping[w])
    print('{} pretrained words added out of {}'.format(n_pretrain, vocab_size))
    return word_embedding, in_doc_word_indices

def update_part_embedding(indices, use_cuda=False):
    """
    update only non-pretrained embeddings
    Args:
        indices: [int]
    """
    indices = torch.LongTensor(indices)
    if use_cuda:
        indices = indices.cuda()
    def hook(grad):
        grad_copy = grad.clone()
        grad_copy[indices] = 0
        return grad_copy
    return hook

        
def init_embedding(input_embedding):
    """
    Initialize embedding
    """
    bias = np.sqrt(3.0 / input_embedding.size(1))
    nn.init.uniform(input_embedding, -bias, bias)

def init_linear(input_linear):
    """
    Initialize linear transformation
    """
    bias = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.weight.size(1)))
    nn.init.uniform(input_linear.weight, -bias, bias)
    if input_linear.bias is not None:
        input_linear.bias.data.zero_()
        
def init_weight(weight):
    nn.init.xavier_uniform(weight)
    
def init_lstm(input_lstm):
    """
    Initialize lstm
    """
    for ind in range(0, input_lstm.num_layers):
        weight = eval('input_lstm.weight_ih_l'+str(ind))
        bias = np.sqrt(6.0 / (weight.size(0)/4 + weight.size(1)))
        nn.init.uniform(weight, -bias, bias)
        weight = eval('input_lstm.weight_hh_l'+str(ind))
        bias = np.sqrt(6.0 / (weight.size(0)/4 + weight.size(1)))
        nn.init.uniform(weight, -bias, bias)
    
    if input_lstm.bias:
        for ind in range(0, input_lstm.num_layers):
            weight = eval('input_lstm.bias_ih_l'+str(ind))
            weight.data.zero_()
            weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
            weight = eval('input_lstm.bias_hh_l'+str(ind))
            weight.data.zero_()
            weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1

def save_checkpoint(state, track_list, filename):
    """
    save checkpoint
    """
    with open(filename + '.json', 'w') as f:
        json.dump(track_list, f)
    torch.save(state, filename + '.model')

def _load_corpus_txt(corpus_dir):
    """
    load tokenized sentences, for training tree models
    """
    ret = []
    filename = 'sent.toks'
    for d in ['train', 'val', 'test']:
        path = os.path.join(corpus_dir, d, filename)
        try:
            with open(path, 'r') as f:
                ret.append([line.strip().split() for line in f.readlines()])
        except Exception as inst:
            print(inst)
            ret.append(None)
        
    train, val, test = ret
    return train, val, test
        
def load_corpus(corpus_dir, ddi=True):
    """
    load ddi corpus
    """
    def decode_int_string(l):
        """
        decode string into interger list
        """
        return [int(c) for c in l.split()]
    def load_file(path):
        def parse_line(line):
            line = line.split('|')
            single_line = ddi2013.SingleLine(*line)
            single_line = single_line._replace(sent=single_line.sent.split(), p1=decode_int_string(single_line.p1), p2=decode_int_string(single_line.p2))            
            return single_line

        corpus = None
        try:
            with open(path, 'r') as f:
                corpus = [parse_line(line) for line in f]
        except Exception as inst:
            print(inst)
        return corpus    
    
    ret = []
    prefices = ['train', 'val', 'test']
    for prefix in prefices:
        path = os.path.join(corpus_dir, prefix + '.ddi')
        ret.append(load_file(path))
    
    if not ddi:
        filename = 'sent.toks'
        
        for i, prefix in enumerate(prefices):
            path = os.path.join(corpus_dir, prefix, filename)
            try:
                with open(path, 'r') as f:
                    for j, line in enumerate(f):
                        ret[i][j] = ret[i][j]._replace(sent=line.strip().split())
                    
            except Exception as inst:
                print(inst)
    
    return tuple(ret)
    
def evaluate(y_true, y_pred, labels=None, target_names=None):
    """
    calculate (micro) precision, recall, and f1 score
    Args:
        labels: [int] labels to consider (e.g., skip false ddi relation)
    """
    average = 'micro'
    def _evaluate(scorer):
        return scorer(y_true, y_pred, labels=labels, average=average)
    precision = _evaluate(precision_score)
    recall = _evaluate(recall_score)
    f1 = _evaluate(f1_score)
#    print(classification_report(y_true, y_pred, labels=labels, target_names=target_names))
    return precision, recall, f1
    
def softmax(input, mask=None, dim=1):
    """
    compute softmax for input; set inf to maskout elements
    input: [d0, d1, d2, ..., d_n-1]
    mask: [d0, 1, 1, d_dim, ..., 1]
    """
    
    if mask is not None:
        input.masked_fill_(mask, -1e12)
    
    return F.softmax(input, dim)

def make_variable(tensor, cuda=False, volatile=False, requires_grad=False):
    if cuda:
        tensor = tensor.cuda()
    return torch.autograd.Variable(tensor, volatile=volatile, requires_grad=requires_grad)

def get_class_weights(l):
    """
    compute class weight according to the following formula
    given C classes with number of instances being n_1, ..., n_C
    obtain class weight w_1, ..., w_C by
    w_1*n_1 = ... = w_C*n_C               (1)
    sum_{i=1}^C w_i*n_i = sum_{i=1}^C n_i (2)
    which yields w_i = sum_{i=1}^C n_i / (C*n_i)
    Args:
        l: [int], containing class labels
    """
    n_tot = len(l)
    counter = Counter(l)
    C = len(counter)
    W = ((c, n_tot*1.0/(C*n)) for c, n in counter.items())
    W = [tup[1] for tup in sorted(W, key=itemgetter(0))]
    
    return W

def build_loss(args, class_weights=None):
    if args.loss == 'crossentropy':
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    elif args.loss == 'marginloss':
        criterion = nn.MultiMarginLoss(p=1, margin=args.margin, size_average=True, weight=class_weights)
    elif args.loss == 'hingeloss':
        criterion = HingeLoss(args)
    else:
        raise ValueError('Unknown loss: {}'.format(args.loss))
    
    return criterion

def build_model(args, vocab_size, tagset_size):
    if args.model == 'InterAttentionLSTM':
        model = InterAttentionLSTM(vocab_size, tagset_size, args)
    elif args.model == 'AttentionPoolingLSTM':
        model = AttentionPoolingLSTM(vocab_size, tagset_size, args)
    elif args.model == 'lstm':
        model = LSTM(vocab_size, tagset_size, args)
    else:
        raise ValueError('Unknown model {}'.format(args.model))
        
    return model

def distance(x, y):
    """
    Args:
        x, y: Tensor/Variable, with broadcastable size
    """
    dist = torch.norm(x - y, p=2)
    
    return dist

def analyze_f1_by_length(pred_tup, t_map):
    # sort by length
    pred_tup = sorted(pred_tup, key=lambda x: x[2])
    pred_buckets = []
    bucket = []
    interval = [30, 100]
    i = 0
    for tup in pred_tup:
        if tup[2] > interval[i]:
            pred_buckets.append((interval[i], bucket))
            i += 1
            bucket = [] 
        bucket.append(tup)
        if i >= len(interval):
            break
    
    ivt_t_map = {v:k for k, v in t_map.items()}
    labels = [k for k,v in ivt_t_map.items() if v != 'null']
    t_names = [ivt_t_map[l] for l in labels]
    f1_by_len = []
    for bucket in pred_buckets:
        l, bucket = bucket
        if len(bucket) == 0:
            print('Length {} does not have instances'.format(l))
        _y_true = [item[0] for item in bucket]
        _y_pred = [item[1] for item in bucket]
        prec, rec, f1 = evaluate(_y_true, _y_pred, labels=labels, target_names=t_names)  
        f1_by_len.append((l, f1))
    return f1_by_len