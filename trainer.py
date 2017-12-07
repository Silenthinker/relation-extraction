#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

import torch
from torch.autograd import Variable
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

import utils

class Trainer:
    
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
        
    
    def _build_optimizer(self):
        if self.args.optimizer == 'sgd':
            return torch.optim.SGD(self.model.parameters(), lr=self.args.lr, 
                                   momentum=self.args.momentum, 
                                   weight_decay=self.args.weight_decay, nesterov=False)
        elif self.args.optimizer == 'nag':
            return torch.optim.SGD(self.model.parameters(), lr=self.args.lr, 
                                   momentum=self.args.momentum, 
                                   weight_decay=self.args.weight_decay, nesterov=True)
        elif self.args.optimizer == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=self.args.lr,
                                    betas=eval(self.args.adam_betas),
                                    weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adagrad':
            return torch.optim.Adagrad(self.model.parameters(), lr=self.args.lr,
                                       weight_decay=self.args.weight_decay)
        else:
            raise ValueError('Unknown optimizer: {}'.format(self.args.optimizer))
            
    
    def _build_lr_scheduler(self):
        if self.args.lr_scheduler == 'rop':
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                      factor=self.args.lr_decay,
                                                                      patience=self.args.patience) # default mode: min
        elif self.args.lr_scheduler == 'lambdalr':
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda epoch: 1. / (1 + (epoch + 1) * self.args.lr_decay))
        else:
            raise ValueError('Unknown lr scheduler: {}'.format(self.args.optimizer))
            
        return lr_scheduler
    
    def get_model(self):
        return self.model
            
    def _forward(self, eval=False):
        if eval:
            self.model.eval()
        else:
            self.model.train()
            self.optimizer.zero_grad()
        
        output, _ = self.model(self._sample['feature'], self._sample['position']) 
        self.loss = self.criterion(output, self._sample['target']) # sum of losses
        
        return output
    
    def _backward_and_opt(self):
        if self.loss is not None:
            avg_loss = self.loss / self._sample['size']
            avg_loss.backward()
        torch.nn.utils.clip_grad_norm(self.model.parameters(), self.args.clip_grad_norm)
        self.optimizer.step()
    
    def train_step(self, sample):
        """
        prepare sample, forward, and backward
        """
        self._prepare_sample(sample, volatile=False, cuda=self.args.cuda)
        
        self._forward()
        
        self._backward_and_opt()
        
        return self.loss
        
    
    def valid_step(self, sample):
        # prepare sample
        self._prepare_sample(sample, volatile=True, cuda=self.args.cuda)
        
        # forward
        output = self._forward(eval=True)
        self.loss = self.criterion(output, self._sample['target'])
        
        return output, self.loss
        
    def _get_lr(self):
        return self.optimizer.param_groups[0]['lr']
    
    def lr_step(self, val_loss=None, epoch=None):
        if self.args.lr_scheduler == 'rop':
            self.lr_scheduler.step(val_loss, epoch)
        else:
            self.lr_scheduler.step(epoch)
        
        return self.optimizer.param_groups[0]['lr']
    
    def _prepare_sample(self, sample, volatile, cuda):
        self._sample = utils.prepare_sample(sample, volatile=volatile, cuda=cuda)
        
        