#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
import torch
from torch import nn
from torch.autograd import Variable

import utils

CRITERION = ['crossentropy', 'marginloss', 'hingeloss']

class HingeLoss(nn.Module):
    
    def __init__(self, args):
        super().__init__()
        self.margin = args.margin
        self.cuda = args.cuda
        
    def forward(self, input, target):
        """
        Args:
            input: [batch_size, num_class]
            target: [batch_size]
        """
        batch_size, num_class = input.size()
        mask = torch.ByteTensor(input.size())
        if self.cuda:
            mask = mask.cuda()
        mask.fill_(1)
        mask.scatter_(1, target.data.view(batch_size, -1), 0)
        pos_scores = torch.masked_select(input, utils.make_variable(~mask, self.cuda, False, requires_grad=False)) # [batch_size]
        neg_scores = torch.masked_select(input, utils.make_variable(mask, self.cuda, False, requires_grad=False)).view(batch_size, -1) # [batch_size, num_class - 1]
        max_neg_scores, _ = torch.max(neg_scores, 1, keepdim=False) # [batch_size]
        _losses = self.margin - pos_scores + max_neg_scores
        losses = torch.max(utils.make_variable(torch.Tensor([0]), self.cuda, False, requires_grad=False), _losses)
        loss = torch.mean(losses)
        
        return loss
    
if __name__ == '__main__':
    criterion = HingeLoss()
    input = torch.Tensor([[0.2, 2, 0.1], [2, 1.2, 0.1]])
    target = torch.LongTensor([1, 0])
    print(criterion(input, target))
        
        
    