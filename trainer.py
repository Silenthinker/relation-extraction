#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
from time import time

import torch
from torch.autograd import Variable
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

from utils import make_variable
from regularizer import Regularizer

class BasicTrainer:
    
    OPTIMIZERS = ['adagrad', 'adam', 'nag', 'sgd']
    LR_SCHEDULER = ['lambdalr', 'rop'] # rop: ReduceLROnPlateau
    
    def __init__(self, args, model, criterion):
        self.args = args
        
        # model and criterion
        self.model = model
        self.criterion = criterion
        
        # cuda
        if self.args.cuda:
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()
        
        # optimizer
        self.optimizer = self._build_optimizer()
        
        # lr scheduler
        self.lr_scheduler = self._build_lr_scheduler()
        
        # regularizer
        self.regularizer = self.__build_regularizer(self.model.reg_params)
        
        model.share_memory()
        
    
    def _build_optimizer(self):
        if self.args.optimizer == 'sgd':
            return torch.optim.SGD(self.model.parameters(), lr=self.args.lr, 
                                   momentum=self.args.momentum, nesterov=False)
        elif self.args.optimizer == 'nag':
            return torch.optim.SGD(self.model.parameters(), lr=self.args.lr, 
                                   momentum=self.args.momentum, nesterov=True)
        elif self.args.optimizer == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=self.args.lr,
                                    betas=eval(self.args.adam_betas))
        elif self.args.optimizer == 'adagrad':
            return torch.optim.Adagrad(self.model.parameters(), lr=self.args.lr)
        else:
            raise ValueError('Unknown optimizer: {}'.format(self.args.optimizer))
            
    def __build_regularizer(self, params):
        return Regularizer(params, self.args.weight_decay, 'l2')
        
    def _build_lr_scheduler(self):
        if self.args.lr_scheduler == 'rop':
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                      factor=self.args.lr_decay,
                                                                      patience=self.args.patience) # default mode: min
        elif self.args.lr_scheduler == 'lambdalr':
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda epoch: 1. / (1 + epoch * self.args.lr_decay))
        else:
            raise ValueError('Unknown lr scheduler: {}'.format(self.args.optimizer))
            
        return lr_scheduler
    
    def get_model(self):
        return self.model
            
    def _forward(self, eval=False):
        raise NotImplementedError("Please implement this method")
    
    def _backward_and_opt(self):
        if self.loss is not None:
            self.loss.backward()
        torch.nn.utils.clip_grad_norm(self.model.parameters(), self.args.clip_grad_norm)
        self.optimizer.step()
    
    def train_step(self, sample):
        """
        prepare sample, forward, and backward
        """
        t = {}
        t0 = time()
        self._prepare_sample(sample, volatile=False, cuda=self.args.cuda)
        t1 = time()
        self._forward()
        t2 = time()
        self._backward_and_opt()
        t3 = time()
        t['prepare'] = t1 - t0
        t['forward'] = t2 - t1
        t['backward'] = t3 - t2
        return self.loss.data[0], t
        
    
    def valid_step(self, sample):
        raise NotImplementedError("Please implement this method")
    
    def pred_step(self, sample):
        raise NotImplementedError("Please implement this method")
        
    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']
    
    def lr_step(self, val_loss=None, epoch=None):
        if self.args.lr_scheduler == 'rop':
            self.lr_scheduler.step(val_loss, epoch)
        else:
            self.lr_scheduler.step(epoch)
        
        return self.optimizer.param_groups[0]['lr']
    
    def _prepare_sample(self, sample, volatile, cuda):
        raise NotImplementedError("Please implement this method")
        
class SeqTrainer(BasicTrainer):
    
    def __init__(self, args, model, criterion):
        super().__init__(args, model, criterion)
        
    def _forward(self, eval=False):
        if eval:
            self.model.eval()
        else:
            self.model.train()
            self.optimizer.zero_grad()
        
        output_dict, _ = self.model(self._sample['feature'], self._sample['position'], self._sample['mask']) 
        self.loss = self.criterion(output_dict['output'], self._sample['target']) # sum of losses
        if self.args.weight_decay > 0:
            self.loss += self.regularizer()
        
        return output_dict
    
    def valid_step(self, sample):
        # prepare sample
        self._prepare_sample(sample, volatile=True, cuda=self.args.cuda)
        
        # forward
        output_dict = self._forward(eval=True)
        self.loss = self.criterion(output_dict['output'], self._sample['target'])
        
        return output_dict, self.loss.data[0]
    
    def pred_step(self, sample):
        self._prepare_sample(sample, volatile=True, cuda=self.args.cuda)
        
        self.model.eval()
        output_dict, pred = self.model.predict(self._sample['feature'], self._sample['position'], self._sample['mask'])
#        print(sample['target'], pred)
        return output_dict, pred
    
    def _prepare_sample(self, sample, volatile, cuda):
        self._sample = {
                'index': make_variable(sample['index'], cuda=False, volatile=volatile),
                'feature': make_variable(sample['feature'], cuda=cuda, volatile=volatile), 
                'position': make_variable(sample['position'], cuda=cuda, volatile=volatile), 
                'target': make_variable(sample['target'], cuda=cuda, volatile=volatile).view(-1),
                'size': len(sample['index']),
                'mask': make_variable(sample['mask'], cuda=cuda, volatile=volatile),
                }

class TreeTrainer(BasicTrainer):
    def __init__(self, args, model, criterion):
        super().__init__(args, model, criterion)
        self.batch_size = args.batch_size
        
    def _forward(self, eval=False):
        if eval:
            self.model.eval()
        else:
            self.model.train()
            self.optimizer.zero_grad()
            
        self.loss = 0
        
        res = {}
        for feature, position, tree, target in zip(self._sample['feature'], self._sample['position'], self._sample['tree'], self._sample['target']):
            # item is dict
            if not self.args.position:
                position = None
            output_dict, _ = self.model(tree, feature, position)
            self.loss += self.criterion(output_dict['output'], target)
            # [dict] to dict[list]
            if not res:
                res = {k:[v.data] for k, v in output_dict.items()}
            else:
                [res[k].append(v.data) for k, v in output_dict.items()]
                
        self.loss /= self.batch_size
        
        if self.args.weight_decay > 0:
            self.loss += self.regularizer()
        
        res['output'] = torch.cat(res['output'], dim=0)
        return res
    
    def valid_step(self, sample):
        # prepare sample
        self._prepare_sample(sample, volatile=True, cuda=self.args.cuda)
        
        # forward
        output_dict = self._forward(eval=True)
        self.loss = self.criterion(make_variable(output_dict['output'], cuda=self.args.cuda, volatile=True), self._sample['target'])
        
        return self.loss.data[0]
    
    def pred_step(self, sample):
        
        self._prepare_sample(sample, volatile=True, cuda=self.args.cuda)
        
        self.model.eval()
        
        res = []
        for feature, position, tree in zip(self._sample['feature'], self._sample['position'], self._sample['tree']):
            _, pred = self.model.predict(tree, feature, position)
            res.append(pred)
#        print(sample['target'], pred)
        return torch.stack(res, dim=0)
    
    def _prepare_sample(self, sample, volatile, cuda):
        """
        Args:
            sample: dict of list
        
        self._sample:
            dict:
                feature: [Var(LongTensor)]
                position: [Var(LongTensor)]
                target: Var(LongTensor)
                tree: [Tree]
        """
        def helper(k):
           sample[k] = [make_variable(item, cuda=cuda, volatile=volatile) for item in sample[k]]
        self._sample = {}
        self._sample['feature'] = [make_variable(item, cuda=cuda, volatile=volatile) for item in sample['feature']]
        self._sample['position'] = [make_variable(item, cuda=cuda, volatile=volatile) for item in sample['position']]
        self._sample['target'] = make_variable(sample['target'], cuda=cuda, volatile=volatile).view(-1)
        self._sample['tree'] = sample['tree']