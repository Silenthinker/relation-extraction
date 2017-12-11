#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import utils
        
class LSTM(nn.Module):
    """LSTM model

    args: 
        vocab_size: size of word dictionary
        tagset_size: size of label set
        embedding_dim: size of word embedding
        hidden_dim: size of word-level blstm hidden dim
        rnn_layers: number of word-level lstm layers
        dropout_ratio: dropout ratio
    """
    
    def __init__(self, vocab_size, tagset_size, args):
        super(LSTM, self).__init__()
        self.embedding_dim = args.embedding_dim
        
        self.position = args.position
        self.position_dim = args.position_dim if args.position else 0
        self.position_size = 2*args.position_bound + 1 # [-position_bound, position_bound]
        self.position_bound = args.position_bound
        
        self.hidden_dim = args.hidden_dim
        self.vocab_size = vocab_size
        self.tagset_size = tagset_size
        self.rnn_layers = args.rnn_layers
        self.dropout_ratio = args.dropout_ratio
        self.args = args
        
        self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim)
        if args.position:
            self.position_embeds = nn.Embedding(self.position_size, self.position_dim)
        
        ## TODO:
    
        self.lstm = nn.LSTM(self.embedding_dim + 2*self.position_dim, self.hidden_dim // 2,
                            num_layers=self.rnn_layers, bidirectional=True, 
                            dropout=self.dropout_ratio, batch_first=True)
        '''
        self.lstm = nn.GRU(self.embedding_dim + 2*self.position_dim, self.hidden_dim // 2,
                            num_layers=self.rnn_layers, bidirectional=True, 
                            dropout=self.dropout_ratio, batch_first=True)
        '''
        self.dropout1 = nn.Dropout(p=self.dropout_ratio)
        self.dropout2 = nn.Dropout(p=self.dropout_ratio)
        self.linear = nn.Linear(self.hidden_dim, tagset_size)
        self.reg_params = []
      
    @property
    def reg_params(self):
        return self.__reg_params
    
    @reg_params.setter
    def reg_params(self, params):
        self.__reg_params = params
    
    def load_pretrained_embedding(self, pre_embeddings):
        """
        load pre-trained word embedding

        args:
            pre_word_embeddings (self.word_size, self.word_dim) : pre-trained embedding
        """
        assert (pre_embeddings.size()[1] == self.embedding_dim)
        self.word_embeds.weight = nn.Parameter(pre_embeddings)

    def rand_init_embedding(self):
        utils.init_embedding(self.word_embeds.weight)
        if self.position:
            utils.init_embedding(self.position_embeds.weight)

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
        utils.init_linear(self.linear)

    def update_part_embedding(self, indices):
        hook = utils.update_part_embedding(indices, self.args.cuda)
        self.word_embeds.weight.register_hook(hook)
        
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
        # last_lstm_out = lstm_out[:, -1, :].contiguous().view(-1, self.hidden_dim) # batch_size, hidden_dim
        last_lstm_out, _ = torch.max(lstm_out, dim=1) # max pooling

        d_lstm_out = self.dropout2(last_lstm_out)
        output = self.linear(d_lstm_out) # output: batch_size, tagset_size
        
        return {'output' :output}, hidden
    
    def predict(self, sentence, position, mask=None):
        output_dict, _ = self.forward(sentence, position, mask)
        _, pred = torch.max(output_dict['output'].data, dim=1)
        
        return pred, output_dict
