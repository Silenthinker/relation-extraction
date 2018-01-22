import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var

import utils
# TODO: binary tree for constituency; weight initialization

# module for childsumtreelstm
class ChildSumTreeLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim):
        super(ChildSumTreeLSTM, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)

    def node_forward(self, inputs, child_c, child_h):
        child_h_sum = torch.sum(child_h, dim=0, keepdim=True)

        iou = self.ioux(inputs) + self.iouh(child_h_sum)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = F.sigmoid(i), F.sigmoid(o), F.tanh(u)

        f = F.sigmoid(
                self.fh(child_h) +
                self.fx(inputs).repeat(len(child_h), 1)
            )
        fc = torch.mul(f, child_c)

        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, F.tanh(c))
        return c, h

    def forward(self, tree, inputs):
        _ = [self.forward(tree.children[idx], inputs) for idx in range(tree.num_children)]

        if tree.num_children == 0:
            child_c = Var(inputs[0].data.new(1, self.mem_dim).fill_(0.))
            child_h = Var(inputs[0].data.new(1, self.mem_dim).fill_(0.))
        else:
            child_c, child_h = zip(* map(lambda x: x.state, tree.children))
            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)

        tree.state = self.node_forward(inputs[tree.idx], child_c, child_h)
        return tree.state

class RelationTreeLSTM(nn.Module):
    def __init__(self, vocab_size, tagset_size, args):
        super().__init__()
        self.embedding_dim = args.embedding_dim
        self.hidden_dim = args.hidden_dim
        self.tagset_size = tagset_size
        self.word_embeds = nn.Embedding(vocab_size, self.embedding_dim)
        self.childsumtreelstm = ChildSumTreeLSTM(self.embedding_dim, self.hidden_dim)
        self.linear = nn.Linear(self.hidden_dim, self.tagset_size)
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
        utils.init_linear(self.linear)
        
    def update_part_embedding(self, indices):
        hook = utils.update_part_embedding(indices, self.args.cuda)
        self.word_embeds.weight.register_hook(hook)
        
    def forward(self, tree, inputs):
        inputs_emb = self.word_embeds(inputs)
        state, hidden = self.childsumtreelstm(tree, inputs_emb)
        output = self.linear(state) # output: tagset_size
        return {'output' :output}, hidden
    
    def predict(self, tree, inputs):
        output_dict, _ = self.forward(tree, inputs)
        _, pred = torch.max(output_dict['output'].data, dim=1)
        # TODO: check dimensionality
        
        return output_dict, pred