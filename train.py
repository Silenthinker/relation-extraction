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

def evaluate(model, data_loader, t_map, cuda=False):
    model.eval()
    y_true = []
    y_pred = []
    for feature, target in chain.from_iterable(data_loader):
        feature = autograd.Variable(feature)
        if cuda:
            feature = feature.cuda()
        output, _ = model(feature)
        _, pred = torch.max(output.data, dim=1)
        if cuda:
            pred = pred.cpu() # cast back to cpu
        y_true.append(target.numpy().tolist())
        y_pred.append(pred.numpy().tolist())
    y_true = list(chain.from_iterable(y_true))
    y_pred = list(chain.from_iterable(y_pred))
    ivt_t_map = {v:k for k, v in t_map.items()}
    labels = range(1, 5)
    t_names = [ivt_t_map[l] for l in labels]
    prec, rec, f1 = utils.evaluate(y_true, y_pred, labels=labels, target_names=t_names)        

    return prec, rec, f1

parser = utils.build_parser()
args = parser.parse_args()
args.cuda = not args.disable_cuda and torch.cuda.is_available()
start_epoch = 0

train_raw_corpus, test_raw_corpus = utils.load_corpus(args.train_path, args.test_path)
if not train_raw_corpus or not test_raw_corpus:
    train_raw_corpus = utils.preprocess_ddi(data_path=args.train_corpus_path, output_path=args.train_path, position=args.position)
    test_raw_corpus = utils.preprocess_ddi(data_path=args.test_corpus_path, output_path=args.test_path, position=args.position)
train_corpus = [(line.sent, line.type) for line in train_raw_corpus]
test_corpus = [(line.sent, line.type) for line in test_raw_corpus]

caseless = args.caseless
batch_size = args.batch_size
num_epoch = args.num_epoch
lr = args.lr
momentum = args.momentum
clip_grad_norm = args.clip_grad_norm
"""
train_corpus = [
    (['This', 'causes', 'an', 'increase', '.'], 'effect'), 
    (['It', 'is', 'recommended', 'to', 'do', '.'], 'int'),
    (['This', 'causes', 'an', 'increase', '.'], 'effect'), 
    (['It', 'is', 'recommended', 'to', 'do', '.'], 'int'),
    (['This', 'causes', 'an', 'increase', '.'], 'effect'), 
    (['It', 'is', 'recommended', 'to', 'do', '.'], 'int'),
    (['This', 'causes', 'an', 'increase', '.'], 'effect'), 
    (['It', 'is', 'recommended', 'to', 'do', '.'], 'int'),
    (['This', 'causes', 'an', 'increase', '.'], 'effect'), 
    (['It', 'is', 'recommended', 'to', 'do', '.'], 'int'),
    (['This', 'causes', 'an', 'increase', '.'], 'effect'), 
    (['It', 'is', 'recommended', 'to', 'do', '.'], 'int'),
    (['This', 'causes', 'an', 'increase', '.'], 'effect'), 
    (['It', 'is', 'recommended', 'to', 'do', '.'], 'int'),
    (['This', 'causes', 'an', 'increase', '.'], 'effect'), 
    (['It', 'is', 'recommended', 'to', 'do', '.'], 'int'),
    (['This', 'causes', 'an', 'increase', '.'], 'effect'), 
    (['It', 'is', 'recommended', 'to', 'do', '.'], 'int'),
    (['This', 'causes', 'an', 'increase', '.'], 'effect'), 
    (['It', 'is', 'recommended', 'to', 'do', '.'], 'int'),
    (['This', 'causes', 'an', 'increase', '.'], 'effect'), 
    (['It', 'is', 'recommended', 'to', 'do', '.'], 'int'),
    (['This', 'causes', 'an', 'increase', '.'], 'effect'), 
    (['It', 'is', 'recommended', 'to', 'do', '.'], 'int'),
    ]
"""
# checkpoint
checkpoint_dir = os.path.dirname(args.checkpoint)
if not os.path.isdir(checkpoint_dir):
    os.mkdir(checkpoint_dir)

# preprocessing
sents = [tup[0] for tup in train_corpus]
feature_mapping = utils.build_vocab(sents, mini_count=5, caseless=caseless)
target_mapping = {'null':0, 'advise':1, 'effect':2, 'mechanism':3, 'int':4}
input_features, input_targets = utils.build_corpus(train_corpus, feature_mapping, target_mapping, caseless)
test_features, test_targets = utils.build_corpus(test_corpus, feature_mapping, target_mapping, caseless)

# train/val split
train_features, train_targets, val_features, val_targets = utils.stratified_shuffle_split(input_features, input_targets, train_size=args.train_size)
train_loader = utils.construct_bucket_dataloader(train_features, train_targets, feature_mapping['PAD'], batch_size, is_train=True)
val_loader = utils.construct_bucket_dataloader(val_features, val_targets, feature_mapping['PAD'], batch_size, is_train=False)
test_loader = utils.construct_bucket_dataloader(test_features, test_targets, feature_mapping['PAD'], batch_size, is_train=False)

# build model
vocab_size = len(feature_mapping)
tagset_size = len(target_mapping)
model = LSTM(vocab_size, tagset_size, args)

# loss and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=feature_mapping['PAD'])
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

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
        pretrained_word_embedding = utils.load_word_embedding(args.emb_file, feature_mapping, args.embedding_dim)
        print(pretrained_word_embedding.size())
        print(vocab_size)
        model.load_pretrained_embedding(pretrained_word_embedding)
    model.rand_init(init_embedding=args.rand_embedding)

if args.cuda:
    model.cuda()
    criterion.cuda()

if os.path.isfile(args.load_checkpoint):
    dev_prec, dev_rec, dev_f1 = evaluate(model, val_loader, target_mapping, cuda=args.cuda)
    test_prec, test_rec, test_f1 = evaluate(model, test_loader, target_mapping, cuda=args.cuda)
    print('checkpoint dev_prec: {:.4f}, dev_rec: {:.4f}, dev_f1: {:.4f}, test_prec: {:.4f}, test_rec: {:.4f}, test_f1: {:.4f}'.format(
        dev_prec, dev_rec, dev_f1, test_prec, test_rec, test_f1))

track_list = []
best_f1 = float('-inf')
patience_count = 0
start_time = time.time()
tot_length = sum(map(lambda t:len(t), train_loader))

for epoch in range(start_epoch, num_epoch):
    epoch_loss = 0
    model.train()
    for feature, target in tqdm(chain.from_iterable(train_loader), desc=' - Tot it {}'.format(tot_length)):
        feature = autograd.Variable(feature)
        target = autograd.Variable(target)
        if args.cuda:
            feature = feature.cuda()
            target = target.cuda()
        model.zero_grad()
        output, _ = model(feature) 
        loss = criterion(output, target.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), clip_grad_norm)
        optimizer.step()
        epoch_loss += loss.data[0]

    # update lr
    utils.adjust_learning_rate(optimizer, args.lr / (1 + (start_epoch + 1) * args.lr_decay))
    epoch_loss /= tot_length
    
    dev_prec, dev_rec, dev_f1 = evaluate(model, val_loader, target_mapping, cuda=args.cuda)
    if dev_f1 > best_f1:
        patience_count = 0
        best_f1 = dev_f1

        test_prec, test_rec, test_f1 = evaluate(model, test_loader, target_mapping, cuda=args.cuda)

        track_list.append({'epoch': epoch, 'loss': epoch_loss, 
            'dev_prec': dev_prec, 'dev_rec': dev_rec, 'dev_f1': dev_f1, 
            'test_prec': test_prec, 'test_rec': test_rec, 'test_f1': test_f1})
        print('epoch: {}, loss: {:.4f}, dev_f1: {:.4f}, test_f1: {:.4f}\tsaving...'.format(epoch, epoch_loss, dev_f1, test_f1))

        try:
            utils.save_checkpoint({
                        'epoch': epoch,
                        'state_dict': model.cpu().state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'f_map': feature_mapping,
                        't_map': target_mapping,
                    }, {'track_list': track_list,
                        'args': vars(args)
                        }, args.checkpoint + '_lstm')
        except Exception as inst:
            print(inst)
    else:
        patience_count += 1
        track_list.append({'epoch': epoch,'loss': epoch_loss, 'dev_prec': dev_prec, 'dev_rec': dev_rec, 'dev_f1': dev_f1})
        print('epoch: {}, loss: {:.4f}, dev_f1: {:.4f}'.format(epoch, epoch_loss, dev_f1))

    print('epoch: {} in {} take: {} s'.format(epoch, args.num_epoch, time.time() - start_time))
    if patience_count >= args.patience:
        break


## TODO: 
# special charOffset ;
# keep drug mentions
# better separation and reusability

