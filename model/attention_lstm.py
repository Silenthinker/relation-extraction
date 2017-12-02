#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jyao
"""

import torch
from torch import nn

from model.lstm import LSTM

from utils import softmax

class Attention_0(nn.Module):
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
        att_weight = softmax(out, dim=1)
        return att_weight

class Attention_1(nn.Module):
    """
    self-attention: 2-layer MLP
    """
    def __init__(self, hidden_dim, att_hidden_dim, num_hops):
        super().__init__()
        self.tanh = nn.Tanh()
        self.w1 = nn.Linear(att_hidden_dim, num_hops, bias=False)
        self.dropout1 = nn.Dropout()
        
    def forward(self, input):
        """
        Args:
            input: batch_size, seq_len, hidden_dim
        Return:
            weight of input
        """
        out = self.tanh(input) # batch_size, seq_len, hidden_dim
        out = self.w1(out) # batch_size, seq_len, num_hops
        att_weight = softmax(out, dim=1)
        return att_weight

 
class AttentionLSTM(LSTM):
    def __init__(self, vocab_size, tagset_size, args):
        super().__init__(vocab_size, tagset_size, args)
        self.attention = Attention_1(args.hidden_dim, args.att_hidden_dim, args.num_hops)
        self.att2out = nn.Linear(args.num_hops*args.hidden_dim, self.tagset_size, bias=True)
        self.dropout3 = nn.Dropout(p=args.dropout_ratio)
        
    def forward(self, sentence, position, hidden=None):
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
        att_weight = self.attention(d_lstm_out) # batch_size, seq_length, num_hops
        att_weight = att_weight.transpose(1, 2) # batch_size, num_hops, seq_length
        sent_repr = torch.matmul(att_weight, lstm_out).view(sentence.size(0), -1) # batch_size, num_hops*hidden_dim
        d_sent_repr = self.dropout3(sent_repr)
        output = self.att2out(d_sent_repr) # output: batch_size, tagset_size
        
        return output, hidden
