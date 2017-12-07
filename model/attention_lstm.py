#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jyao
"""

import torch
from torch import nn

from model.lstm import LSTM

import utils
from utils import softmax


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
        att_weight = softmax(out, dim=1)
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

class InterAttention(nn.Module):
    """
    attention to hidden state guided by relation embedding
    """
    def __init__(self, num_embeddings, embedding_dim, input_dim, diagonal):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.input_dim = input_dim
        
        self.relation_embeds = nn.Parameter(torch.Tensor(embedding_dim, num_embeddings))
        self.diagonal = diagonal
        if diagonal:
            assert embedding_dim == input_dim, 'embedding_dim == input_dim for diagonal bilinear matrix'
            self.bilinear = nn.Parameter(torch.Tensor(1, input_dim))
        else:
            self.bilinear = nn.Parameter(torch.Tensor(input_dim, embedding_dim))
    
    def rand_init(self):
        utils.init_weight(self.bilinear)
        utils.init_weight(self.relation_embeds)
    
    def forward(self, input):
        """
        Args:
            input: [batch_size, seq_length, hidden_dim]
        """
        batch_size = input.size(0)
        input = input.contiguous().view(-1, self.input_dim) # [batch_size*seq_length, hidden_dim]
        if self.diagonal:
            out = input*self.bilinear # [batch_size*seq_length, hidden_dim]
        else:
            out = torch.mm(input, self.bilinear) # [batch_size*seq_length, embedding_dim]
        out = torch.mm(out, self.relation_embeds) # [batch_size*seq_length, num_embeddings]
        out = out.view(batch_size, -1, self.num_embeddings)
        out = softmax(out, dim=1)
        
        return out
 
class AttentionPoolingLSTM(LSTM):
    def __init__(self, vocab_size, tagset_size, args):
        super().__init__(vocab_size, tagset_size, args)
        self.attention = SmallIntraAttention(args.hidden_dim, args.att_hidden_dim, args.num_hops)
        self.att2out = nn.Linear(args.num_hops*args.hidden_dim, self.tagset_size, bias=True)
        self.dropout3 = nn.Dropout(p=args.dropout_ratio)
        
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
        ## TODO: dropout or not?
        d_lstm_out = self.dropout2(lstm_out)
#        d_lstm_out = lstm_out
        att_weight = self.attention(d_lstm_out) # batch_size, seq_length, num_hops
        att_weight = att_weight.transpose(1, 2) # batch_size, num_hops, seq_length
        sent_repr = torch.matmul(att_weight, lstm_out).view(sentence.size(0), -1) # batch_size, num_hops*hidden_dim
        d_sent_repr = self.dropout3(sent_repr)
        output = self.att2out(d_sent_repr) # output: batch_size, tagset_size
        
        return output, hidden

class InterAttentionLSTM(LSTM):
    def __init__(self, vocab_size, tagset_size, args):
        super().__init__(vocab_size, tagset_size, args)
        self.attention = InterAttention(tagset_size, args.relation_dim, args.hidden_dim, args.diagonal)
        if args.sent_repr == 'concat':
            self.att2out = nn.Linear(args.hidden_dim*tagset_size, self.tagset_size, bias=True)
        else:
            self.att2out = nn.Linear(args.hidden_dim, self.tagset_size, bias=True)
        self.dropout3 = nn.Dropout(p=args.dropout_ratio)
        
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
        att_weight = self.attention(d_lstm_out) # batch_size, seq_length, tagset_size
        att_weight = att_weight.transpose(1, 2) # batch_size, tagset_size, seq_length
        if self.args.sent_repr == 'concat':
            sent_repr = torch.matmul(att_weight, d_lstm_out).view(sentence.size(0), -1) # batch_size, tagset_size*hidden_dim
        else:
            sent_repr, _ = torch.max(torch.matmul(att_weight, d_lstm_out), dim=1) # [batch_size, hidden_dim]
        d_sent_repr = self.dropout3(sent_repr)
        output = self.att2out(d_sent_repr) # output: batch_size, tagset_size
        
        return output, hidden