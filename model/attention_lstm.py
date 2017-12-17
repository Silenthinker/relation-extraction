#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jyao
"""

import torch
from torch import nn
from torch.nn.functional import normalize

from model.lstm import LSTM

import utils

class LargeIntraAttention(nn.Module):
    """
    self-attention: 2-layer MLP
    """
    def __init__(self, hidden_dim, att_hidden_dim, num_hops):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, att_hidden_dim, bias=False)
        self.dropout1 = nn.Dropout()
        self.dropout2 = nn.Dropout()
        self.tanh = nn.Tanh()
        self.w2 = nn.Linear(att_hidden_dim, num_hops, bias=False)
     
    def rand_init(self):
        pass
    
    def forward(self, input):
        """
        Args:
            input: batch_size, seq_len, hidden_dim
        Return:
            weight of input
        """
        out = self.w1(input) # batch_size, seq_len, att_hidden_dim
        out = self.tanh(out) # batch_size, seq_len, att_hidden_dim
        out = self.dropout1(out)
        out = self.w2(out) # batch_size, seq_len, num_hops
        out = self.dropout2(out)
        att_weight = utils.softmax(out, dim=1)
        return att_weight

class SmallIntraAttention(nn.Module):
    """
    self-attention: 2-layer MLP
    """
    def __init__(self, hidden_dim, att_hidden_dim, num_hops):
        super().__init__()
        self.tanh = nn.Tanh()
        self.w1 = nn.Linear(hidden_dim, num_hops, bias=False)
        self.dropout1 = nn.Dropout()
     
    def rand_init(self):
        utils.init_linear(self.w1)
        
    def forward(self, input, mask=None):
        """
        Args:
            input: batch_size, seq_len, hidden_dim
            mask: None or [batch_size, seq_len]
        Return:
            weight of input
        """
        if mask is not None:
            neg_mask = mask.clone()
            neg_mask.data = ~neg_mask.data
        out = self.tanh(input) # batch_size, seq_len, hidden_dim
        out = self.w1(out) # batch_size, seq_len, num_hops
        att_weight = utils.softmax(out, mask=neg_mask.view(*neg_mask.size(), 1), dim=1)
        return att_weight

class InterAttention(nn.Module):
    """
    attention to hidden state guided by relation embedding
    """
    def __init__(self, dim):
        super().__init__()
        self.w = nn.Parameter(torch.Tensor(1, dim)) # parametrise diagonal matrix
    
    def rand_init(self):
        utils.init_weight(self.w)
        
    def forward(self, input_1, input_2, mask=None):
        """
        Args:
            input_1: [batch_size, seq_length, hidden_dim]
            input_2: [hidden_dim, tagset_size]
        """
        if mask is not None:
            neg_mask = mask.clone()
            neg_mask.data = ~neg_mask.data
        batch_size, _, hidden_dim = input_1.size()
        tagset_size = input_2.size(1)
        input_1 = input_1.contiguous().view(-1, hidden_dim) # [batch_size*seq_length, hidden_dim]
        out = input_1 * self.w
        out = torch.mm(out, input_2) # [batch_size*seq_length, tag_size]
        out = out.view(batch_size, -1, tagset_size)
        out = utils.softmax(out, mask=neg_mask.view(*neg_mask.size(), 1), dim=1)
        
        return out
 
class AttentionPoolingLSTM(LSTM):
    def __init__(self, vocab_size, tagset_size, args):
        super().__init__(vocab_size, tagset_size, args)
        self.attention = SmallIntraAttention(args.hidden_dim, args.att_hidden_dim, args.num_hops)
        self.att2out = nn.Linear(args.num_hops*args.hidden_dim, self.tagset_size, bias=True)
        self.dropout3 = nn.Dropout(p=args.dropout_ratio)
        self.reg_params = [self.word_embeds, self.attention.w1, self.att2out]
        
    def rand_init(self, init_embedding=False):
        """
        random initialization

        args:
            init_embedding: random initialize word embedding or not
        """
        if init_embedding:
            utils.init_embedding(self.word_embeds.weight)
        if self.position:
            utils.init_embedding(self.position_embeds.weight)
        utils.init_lstm(self.lstm)
        utils.init_linear(self.att2out)
        self.attention.rand_init()
        self.att_weight = None
        
    def forward(self, sentence, position, mask=None, hidden=None):
        '''
        args:
            sentence (batch_size, word_seq_len) : word-level representation of sentence
            hidden: initial hidden state
            mask: None or [batch_size, seq_len]
        return:
            output (batch_size, tag_size), hidden
        '''
        w_embeds = self.word_embeds(sentence) # batch_size, seq_len, embedding_dim
        if self.position:
            p_embeds = self.position_embeds(position + self.position_bound) # batch_size, 2*seq_len, p_embed_dim
            p_embeds = p_embeds.view(p_embeds.size(0), p_embeds.size(1) // 2, -1)
            embeds = torch.cat([w_embeds, p_embeds], dim=2)
        else:
            embeds = w_embeds
        d_embeds = self.dropout1(embeds)

        lstm_out, hidden = self.lstm(d_embeds, hidden) # lstm_out: batch_size, seq_length, hidden_dim
        ## TODO: dropout or not?
        d_lstm_out = self.dropout2(lstm_out)
#        d_lstm_out = lstm_out
        att_weight = self.attention(d_lstm_out, mask) # batch_size, seq_length, num_hops
        att_weight = att_weight.transpose(1, 2) # batch_size, num_hops, seq_length
        self.att_weight = att_weight[:, 0, :]
        sent_repr = torch.matmul(att_weight, lstm_out).view(sentence.size(0), -1) # batch_size, num_hops*hidden_dim
        d_sent_repr = self.dropout3(sent_repr)
        output = self.att2out(d_sent_repr) # output: batch_size, tagset_size
        
        return {'output': output, 'att_weight': self.att_weight}, hidden

class InterAttentionLSTM(LSTM):
    def __init__(self, vocab_size, tagset_size, args):
        super().__init__(vocab_size, tagset_size, args)
        self.attention = InterAttention(args.hidden_dim)
        self.tanh = nn.Tanh()
        if args.sent_repr == 'concat':
            self.att2out = nn.Linear(args.hidden_dim*tagset_size, self.tagset_size, bias=True)
        else:
            self.att2out = nn.Linear(args.hidden_dim, self.tagset_size, bias=True)
        self.dropout3 = nn.Dropout(p=args.dropout_ratio)
        self.relation_embeds = nn.Parameter(torch.Tensor(args.hidden_dim, self.tagset_size)) 
        self.reg_params = [self.word_embeds, self.relation_embeds, self.att2out]
        
    def rand_init(self, init_embedding=False):
        """
        random initialization

        args:
            init_embedding: random initialize word embedding or not
        """
        if init_embedding:
            utils.init_embedding(self.word_embeds.weight)
        if self.position:
            utils.init_embedding(self.position_embeds.weight)
        utils.init_lstm(self.lstm)
        utils.init_linear(self.att2out)
        utils.init_weight(self.relation_embeds)
        self.attention.rand_init()
        self.att_weight = None
        
    def forward(self, sentence, position, mask=None, hidden=None):
        '''
        args:
            sentence (batch_size, word_seq_len) : word-level representation of sentence
            hidden: initial hidden state

        return:
            output (batch_size, tag_size), hidden
        '''
        w_embeds = self.word_embeds(sentence) # batch_size, seq_len, embedding_dim
        if self.position:
            p_embeds = self.position_embeds(position + self.position_bound) # batch_size, 2*seq_len, p_embed_dim
            p_embeds = p_embeds.view(p_embeds.size(0), p_embeds.size(1) // 2, -1)
            embeds = torch.cat([w_embeds, p_embeds], dim=2)
        else:
            embeds = w_embeds
        d_embeds = self.dropout1(embeds)

        lstm_out, hidden = self.lstm(d_embeds, hidden) # lstm_out: batch_size, seq_length, hidden_dim
        d_lstm_out = self.dropout2(lstm_out)
        att_weight = self.attention(d_lstm_out, self.relation_embeds, mask) # batch_size, seq_length, tagset_size
        att_weight = att_weight.transpose(1, 2) # batch_size, tagset_size, seq_length
        self.att_weight = att_weight
        if self.args.sent_repr == 'concat':
            sent_repr = torch.matmul(att_weight, lstm_out).view(sentence.size(0), -1) # batch_size, tagset_size*hidden_dim
        else:
            sent_repr, _ = torch.max(torch.matmul(att_weight, lstm_out), dim=1) # [batch_size, hidden_dim]
        d_sent_repr = self.dropout3(sent_repr)
#        output = self.att2out(d_sent_repr) # output: batch_size, tagset_size
#        n_d_sent_repr = normalize(d_sent_repr, dim=1) # [batch_size, hidden_dim]
#        n_relation_embs = normalize(self.relation_embeds, dim=0) # [hidden_dim, tagset_size]
        output = torch.mm(d_sent_repr, self.relation_embeds)

        return {'output': output, 'att_weight': self.att_weight}, hidden
    
        