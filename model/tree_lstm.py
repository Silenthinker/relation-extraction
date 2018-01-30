import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var

import utils
# TODO: binary tree for constituency; weight initialization

# module for childsumtreelstm
class ChildSumTreeLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim):
        super().__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim, bias=True) # ioux is reponsible for bias
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim, bias=False)
        self.fx = nn.Linear(self.in_dim, self.mem_dim, bias=True) # responsible for bias
        self.fh = nn.Linear(self.mem_dim, self.mem_dim, bias=False)
        self.reg_params = [self.ioux, self.iouh, self.fx, self.fh]

    def node_forward(self, inputs, child_c, child_h):
        child_h_sum = torch.sum(child_h, dim=0, keepdim=True)

        iou = self.ioux(inputs) + self.iouh(child_h_sum)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = F.sigmoid(i), F.sigmoid(o), F.tanh(u)

        f = F.sigmoid(
                self.fh(child_h) +
                self.fx(inputs).repeat(len(child_h), 1)
            ) # [num_children, mem_dim]
        fc = torch.mul(f, child_c)

        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, F.tanh(c))
        return c, h

    def forward(self, tree, inputs):
        for node in tree:
            if node.num_children == 0:
                child_c = Var(inputs[0].data.new(1, self.mem_dim).fill_(0.))
                child_h = Var(inputs[0].data.new(1, self.mem_dim).fill_(0.))
            else:
                child_c, child_h = zip(* map(lambda x: x.state, node.children))
                child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0) # child_c, child_h: [num_children, mem_dim]
    
            node.state = self.node_forward(inputs[node.idx], child_c, child_h)
        return tree[-1].state

# Module for binary tree lstm 
class BinaryTreeLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim):
        """
        Args:
            bf: branching factor, int
        """
        super().__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.bf = 2
        
        self.fioux = nn.Linear(self.in_dim, 4 * self.mem_dim, bias=True) # responsible for bias
        self.iouh = nn.Linear(self.bf * self.mem_dim, self.mem_dim * 3, bias=False)
        self.fh = nn.Linear(self.bf * self.mem_dim, self.bf * self.mem_dim, bias=False)
        self.reg_params = [self.fioux, self.iouh, self.fh]
        
    def node_forward(self, inputs, child_c, child_h):
        """
        Args:
            inputs: FloatTensor [in_dim]
            child_c, child_h: FloatTensor [1, bf * mem_dim]
        Returns:
            c, h: FloatTensor [1, mem_dim]
        """
        inputs = inputs.view(1, -1)
        fioux = self.fioux(inputs) # [1, 4*mem_dim]
        fx, ix, ox, ux = torch.split(fioux, fioux.size(1) // 4, dim=1) # [1, mem_dim]
        iouh = self.iouh(child_h)
        ih, oh, uh = torch.split(iouh, iouh.size(1) // 3, dim=1) # [1, mem_dim]
        i, o, u = F.sigmoid(ix + ih), F.sigmoid(ox + oh), F.tanh(ux + uh)
        
        fh = self.fh(child_h) # [1, bf*mem_dim]
        f = F.sigmoid(fx.repeat(1, self.bf) + fh)
        
        fc = torch.mul(f, child_c).view(-1, self.mem_dim) # [bf, mem_dim]

        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, F.tanh(c))
        return c, h        
        
    def forward(self, tree, inputs):
        """
        Args:
            tree: level order traversal
        """
        def _zeros(dim):
            return Var(inputs[0].data.new(1, dim).fill_(0.))
        
        def _initialize_child_tensor(left=None, right=None):
            """
            Initialize tensor
            """
            if left is None:
                left = _zeros(self.mem_dim)
            if right is None:
                right = _zeros(self.mem_dim)
                
            return torch.cat([left, right], dim=1)

        for node in tree:
            left_c, left_h = None, None
            right_c, right_h = None, None
            
            x = _zeros(self.in_dim) if node.num_children > 0 else inputs[node.idx] # input is zero only if node is internal
            
            if node.num_children >= 1:
                left_c, left_h = node.children[0].state
            if node.num_children == 2:
                right_c, right_h = node.children[1].state            
            
            child_c = _initialize_child_tensor(left_c, right_c)
            child_h = _initialize_child_tensor(left_h, right_h)
    
            node.state = self.node_forward(x, child_c, child_h)
        
        return tree[-1].state
    
class RelationTreeLSTM(nn.Module):
    def __init__(self, vocab_size, tagset_size, args):
        super().__init__()
        self.args = args
        self.dropout_ratio = args.dropout_ratio
        self.embedding_dim = args.embedding_dim
        self.position = args.position
        self.position_dim = args.position_dim if self.position else 0
        self.position_bound = args.position_bound
        self.position_size = 2*args.position_bound + 1
        self.hidden_dim = args.hidden_dim
        self.tagset_size = tagset_size
        self.word_embeds = nn.Embedding(vocab_size, self.embedding_dim)
        
        if self.position:
            self.position_embeds = nn.Embedding(self.position_size, self.position_dim)
            
        if args.childsum_tree:
            self.treelstm = ChildSumTreeLSTM(self.embedding_dim + 2*self.position_dim, self.hidden_dim)
        else:
            self.treelstm = BinaryTreeLSTM(self.embedding_dim + 2*self.position_dim, self.hidden_dim)
            
        self.linear = nn.Linear(self.hidden_dim, self.tagset_size)
        self.dropout1 = nn.Dropout(p=self.dropout_ratio)
        self.dropout2 = nn.Dropout(p=self.dropout_ratio)
        self.reg_params = [self.linear] + self.treelstm.reg_params
      
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
            self.rand_init_embedding()
        utils.init_linear(self.linear)
        
    def update_part_embedding(self, indices):
        hook = utils.update_part_embedding(indices, self.args.cuda)
        self.word_embeds.weight.register_hook(hook)
        
    def forward(self, tree, inputs, pos=None):
        inputs_emb = self.word_embeds(inputs) # [seq_len, w_emb_dim]
        if self.position:
            assert pos is not None
            position_emb = self.position_embeds(pos + self.position_bound) # 2*seq_len, p_embed_dim
            position_emb = torch.cat([position_emb[0:position_emb.size(0)//2], position_emb[position_emb.size(0)//2:]], dim=1)
            inputs_emb = torch.cat([inputs_emb, position_emb], dim=1)
            
        d_inputs_emb = self.dropout1(inputs_emb)
        state, hidden = self.treelstm(tree, d_inputs_emb)
        d_state = self.dropout2(state)
        output = self.linear(d_state) # output: tagset_size
        return {'output' :output}, hidden
    
    def predict(self, tree, inputs, pos=None):
        output_dict, _ = self.forward(tree, inputs, pos)
        _, pred = torch.max(output_dict['output'].data, dim=1)
        # TODO: check dimensionality
        
        return output_dict, pred