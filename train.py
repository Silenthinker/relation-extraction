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
from data import DDI2013Dataset
from trainer import Trainer
from model.lstm import LSTM
from model.attention_lstm import AttentionPoolingLSTM, InterAttentionLSTM

def evaluate(trainer, data_loader, t_map, cuda=False):
    y_true = []
    y_pred = []
    tot_length = sum(map(lambda t:len(t), data_loader))
    tot_loss = 0
    for sample in chain.from_iterable(data_loader):
        target = sample['target']
        output, loss = trainer.valid_step(sample)
        _, pred = torch.max(output.data, dim=1)
        if cuda:
            pred = pred.cpu() # cast back to cpu
            loss = loss.cpu()
        tot_loss += loss.data[0]
        y_true.append(target.numpy().tolist())
        y_pred.append(pred.numpy().tolist())
    y_true = list(chain.from_iterable(y_true))
    y_pred = list(chain.from_iterable(y_pred))
    ivt_t_map = {v:k for k, v in t_map.items()}
    labels = [k for k,v in ivt_t_map.items() if v != 'null']
    t_names = [ivt_t_map[l] for l in labels]
    prec, rec, f1 = utils.evaluate(y_true, y_pred, labels=labels, target_names=t_names)        
    avg_loss = tot_loss / tot_length
    return prec, rec, f1, avg_loss

def filterTrueTypes(corpus):
    """
    exclude false interaction type
    """
    return list(filter(lambda tup:tup[1] != 'null', corpus))

def multi2bin(corpus):
    """
    turn multi-class classification to binary one
    """
    return [(tup[0], tup[1] if tup[1] == 'null' else 'true', tup[2], tup[3])for tup in corpus]
    
    
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
    ##TODO: progress bar
    
    # checkpoint
    checkpoint_dir = os.path.dirname(args.checkpoint)
    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    
    # load dataset
    train_raw_corpus, test_raw_corpus = utils.load_corpus(args.train_path, args.test_path)
    if not train_raw_corpus or not test_raw_corpus:
        train_raw_corpus = utils.preprocess_ddi(data_path=args.train_corpus_path, output_path=args.train_path, position=True)
        test_raw_corpus = utils.preprocess_ddi(data_path=args.test_corpus_path, output_path=args.test_path, position=True)
    train_corpus = [(line.sent, line.type, line.p1, line.p2) for line in train_raw_corpus]
    test_corpus = [(line.sent, line.type, line.p1, line.p2) for line in test_raw_corpus]
    
    ##
#    train_corpus = multi2bin(train_corpus)
#    test_corpus = multi2bin(test_corpus)
    
    start_epoch = 0
    caseless = args.caseless
    batch_size = args.batch_size
    num_epoch = args.num_epoch
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
    
    
    # preprocessing
    sents = [tup[0] for tup in train_corpus]
    feature_map = utils.build_vocab(sents, min_count=args.min_count, caseless=caseless)
    ##
#    target_map = {c:i for i, c in enumerate(['null', 'true'])}
    target_map = DDI2013Dataset.target_map
    input_features, input_targets = utils.build_corpus(train_corpus, feature_map, target_map, caseless)
    test_features, test_targets = utils.build_corpus(test_corpus, feature_map, target_map, caseless)
    
    # train/val split
    train_features, train_targets, val_features, val_targets = utils.stratified_shuffle_split(input_features, input_targets, train_size=args.train_size)
    class_weights = torch.Tensor(utils.get_class_weights(train_targets)) if args.class_weight else None
    train_loader = utils.construct_bucket_dataloader(train_features, train_targets, feature_map['PAD'], batch_size, args.position_bound, is_train=True)
    val_loader = utils.construct_bucket_dataloader(val_features, val_targets, feature_map['PAD'], batch_size, args.position_bound, is_train=False)
    test_loader = utils.construct_bucket_dataloader(test_features, test_targets, feature_map['PAD'], batch_size, args.position_bound, is_train=False)
    print('Preprocessing done! Vocab size: {}'.format(len(feature_map)))
    
    # build model
    vocab_size = len(feature_map)
    tagset_size = len(target_map)
    model = InterAttentionLSTM(vocab_size, tagset_size, args) if args.attention else LSTM(vocab_size, tagset_size, args)
    
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
    
    # trainer
    trainer = Trainer(args, model, criterion)
    
    if os.path.isfile(args.load_checkpoint):
        dev_prec, dev_rec, dev_f1, _ = evaluate(trainer, val_loader, target_map, cuda=args.cuda)
        test_prec, test_rec, test_f1, _ = evaluate(trainer, test_loader, target_map, cuda=args.cuda)
        print('checkpoint dev_prec: {:.4f}, dev_rec: {:.4f}, dev_f1: {:.4f}, test_prec: {:.4f}, test_rec: {:.4f}, test_f1: {:.4f}'.format(
            dev_prec, dev_rec, dev_f1, test_prec, test_rec, test_f1))
    
    track_list = []
    best_f1 = float('-inf')
    patience_count = 0
    start_time = time.time()
    tot_length = sum(map(lambda t:len(t), train_loader))
    
    for epoch in range(start_epoch, num_epoch):
        epoch_loss = 0
        for sample in tqdm(chain.from_iterable(train_loader), desc=' - Tot it {}'.format(tot_length)):
            loss = trainer.train_step(sample)
            epoch_loss += loss.data[0]
    
        # update lr
        trainer.lr_step()
        
        epoch_loss /= tot_length
        
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

if __name__ == '__main__':
    main()
## TODO: 
# residual connection using rnncell
# add regularization for self-attention
# keep drug mentions
