import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var

import utils

'''
class IntraAttention(nn.Module):
    """
    self-attention: 1-layer MLP
    """
    def __init__(self, in_dim, att_dim):
        super().__init__()
        self.w1 = nn.Linear(in_dim, att_dim, bias=False)
        self.w2 = nn.Linear(att_dim, 1, bias=False)
        self.dropout = nn.Dropout()
        
        self.reg_params = [self.w1, self.w2]
     
    def rand_init(self):
        utils.init_linear(self.w1)
        utils.init_linear(self.w2)
        
    def forward(self, inputs):
        """
        Args:
            inputs: [N, in_dim]
        Return:
            weight of input: [N, 1]
        """
        out = self.w1(self.dropout(inputs)) # [N, att_dim]
        out = self.w2(F.tanh(out)) # [N, 1]
        att_weight = F.softmax(out, dim=0)
        
        return att_weight
'''

class StructuralAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.w = nn.Linear(in_dim, 1, bias=False)
        
        self.reg_params = [self.w]
    
    def rand_init(self):
        utils.init_linear(self.w)
        
    def forward(self, tree, inputs):
        """
        Args:
            tree: [Tree]
            inputs: [N, in_dim]
        Return:
            weight of input: [N, 1]
            attentional states: Tensor[N, dim]
        """
        out = self.w(F.tanh(inputs)) # [N, 1]
        out = torch.exp(out) # exponential
        root = tree[-1]
        for i, node in enumerate(tree):
            node.val = out[i]
        root.val.data.fill_(1) # root node always has score equal to 1
        
        # renormalize scores
        def _renormalize(root):
            if root.num_children == 2:
                sum_score = root.children[0].val + root.children[1].val + 1e-10
                multiplier = 1 / sum_score * root.val
                for c in root.children:
                    c.val = c.val * multiplier
                    _renormalize(c)
            elif root.num_children == 1:
                c = root.children[0]
                c.val = root.val
                _renormalize(c)
            elif root.num_children == 0:
                return
        
        _renormalize(root)
        
        # compute attentional hidden states
        
        for node in tree:
            if node.num_children == 2:
                left_c, left_h = node.children[0].state
                right_c, right_h = node.children[1].state
                left_s = node.children[0].val
                right_s = node.children[1].val   
                node.state = (left_c * left_s + right_c * right_s, left_h * left_s + right_h * right_s)
    
        att_scores = torch.stack([node.val for node in tree], dim=0)   
        return att_scores, [node.state for node in tree]

class InterAttention(nn.Module):
    """
    attention to hidden state guided by relation embedding
    """
    def __init__(self, dim):
        super().__init__()
        self.W = nn.Linear(dim, dim, bias=False)
        
        self.reg_params = [self.W]
    
    def rand_init(self):
        utils.init_linear(self.W)
        
    def forward(self, input_1, input_2):
        """
        Args:
            input_1: [seq_length, hidden_dim]
            input_2: [hidden_dim, tagset_size]
        Returns:
            out: [seq_length, tag_size]
        """
        
        # H W E
        out = self.W(input_1) # [seq_length, hidden_dim]
        out = torch.mm(input_1, input_2) # [seq_length, tag_size]
        
        
        out = F.softmax(out, dim=0)
        
        return out
    
class IntraAttention(nn.Module):
    """
    self-attention: 1-layer MLP
    """
    def __init__(self, in_dim):
        super().__init__()
        self.w = nn.Linear(in_dim, 1, bias=False)
        self.dropout = nn.Dropout()
        
        self.reg_params = [self.w]
     
    def rand_init(self):
        utils.init_linear(self.w)
        
    def forward(self, inputs):
        """
        Args:
            inputs: [N, in_dim]
        Return:
            weight of input: [N, 1]
        """
        out = self.w(F.tanh(self.dropout(inputs))) # [N, 1]
        att_weight = F.softmax(out, dim=0)
        
        return att_weight    
    
class TreeRNNBase(nn.Module):
    def __init__(self, in_dim, mem_dim, dropout=0):
        super().__init__()
        
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.dropout_ratio = dropout
        
        self.forward_dropout = nn.Dropout(p=self.dropout_ratio)
        self.semeniuta_dropout = nn.Dropout(p=self.dropout_ratio)
    
    def rand_init(self):
        raise NotImplementedError
        
    def node_forward(self, *args):
        raise NotImplementedError
        
    def forward(self, tree, inputs):
        raise NotImplementedError
        

class BinaryTreeGRU(TreeRNNBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.bf = 2
        
        self.grzx = nn.Linear(self.in_dim, 3 * self.mem_dim, bias=True)
        self.rzh = nn.Linear(self.bf * self.mem_dim, 2 * self.bf * self.mem_dim, bias=False)
        self.gh = nn.Linear(self.mem_dim, self.mem_dim, bias=False)
        
        self.reg_params = [self.grzx, self.rzh, self.gh]
        
    def rand_init(self):
        # initialize weight
        for p in [self.grzx, self.rzh, self.gh]:
            utils.init_weight(p.weight)
        
        # initialize forget gate bias
        self.grzx.bias.data.zero_()
        self.grzx.bias.data[self.mem_dim:] = 1 # bias for z and r gate is init to 1
        
    def node_forward(self, inputs, child_h):
        """
        Args:
            inputs: FloatTensor [in_dim]
            child_h: FloatTensor [1, 2 * mem_dim]
        Returns:
            h: FloatTensor [1, mem_dim]
        """
        
        inputs = inputs.view(1, -1) # [1, in_dim]
        d_inputs = self.forward_dropout(inputs)
        
        grzx = self.grzx(d_inputs)
        gx, rzx = grzx[:, :self.mem_dim], grzx[:, self.mem_dim:] # [1, mem_dim], [1, 2*mem_dim]
        
        rzh = self.rzh(child_h) # [1, 4*mem_dim]
        rz = F.sigmoid(rzx.repeat(1, self.bf) + rzh)
        r, z = torch.split(rz, self.bf * self.mem_dim, dim=1)
        r, z = r.view(self.bf, -1), z.view(self.bf, -1)
        
        child_h = child_h.view(self.bf, -1)
        
        g = F.tanh(gx + self.gh(torch.sum(torch.mul(r, child_h), dim=0))) # [1, mem_dim]
        
        h = torch.sum(torch.mul(z, child_h), dim=0) + torch.mul((1 - torch.sum(z, dim=0) / self.bf), self.semeniuta_dropout(g))
        
        return h
        
    def forward(self, tree, inputs):
        
        def _zeros(dim):
            return Var(inputs[0].data.new(1, dim).fill_(0.))
        
        def _initialize_child_tensor(left=None, right=None):
            """
            Initialize tensor
            Return:
                [Tensor], [1, mem_dim]
            """
            if left is None:
                left = _zeros(self.mem_dim)
            if right is None:
                right = _zeros(self.mem_dim)
                
            return torch.cat([left, right], dim=1)
        
        for node in tree:
            left_h, right_h = None, None
            
            x = _zeros(self.in_dim) if node.num_children > 0 else inputs[node.idx] # input is zero only if node is internal
            
            if node.num_children >= 1:
                left_h = node.children[0].state
            if node.num_children == 2:
                right_h = node.children[1].state            
        
            child_h = _initialize_child_tensor(left_h, right_h)
    
            node.state = self.node_forward(x, child_h)
        
        return [(node.state, node.state) for node in tree]
        
# module for childsumtreelstm
class ChildSumTreeLSTM(TreeRNNBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim, bias=True) # ioux is reponsible for bias
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim, bias=False)
        
        self.fx = nn.Linear(self.in_dim, self.mem_dim, bias=True) # responsible for bias
        self.fh = nn.Linear(self.mem_dim, self.mem_dim, bias=False)
        
        # self.reg_params = [self.ioux, self.iouh, self.fx, self.fh]
        self.reg_params = []

    def rand_init(self):
        """
        Initialize
        """
        # initialize weights
        for p in [self.ioux, self.iouh, self.fx, self.fh]:
            utils.init_weight(p.weight)
        
        # initialize forget gate bias
        self.ioux.bias.data.zero_()
        self.fx.bias.data[:] = 1
        
    def node_forward(self, inputs, child_c, child_h):
        child_h_sum = torch.sum(child_h, dim=0, keepdim=True)
        
        d_inputs = self.forward_dropout(inputs) # apply forward dropout
        
        iou = self.ioux(d_inputs) + self.iouh(child_h_sum)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = F.sigmoid(i), F.sigmoid(o), F.tanh(u)

        f = F.sigmoid(
                self.fh(child_h) +
                self.fx(d_inputs).repeat(len(child_h), 1)
            ) # [num_children, mem_dim]
        fc = torch.mul(f, child_c)

        c = torch.mul(i, self.semeniuta_dropout(u)) + torch.sum(fc, dim=0, keepdim=True)
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
            
        return [node.state for node in tree]

# Module for binary tree lstm 
class BinaryTreeLSTM(TreeRNNBase):
    def __init__(self, *args, **kwargs):
        """
        Args:
            bf: branching factor, int
        """
        super().__init__(*args, **kwargs)
        
        self.bf = 2
        
        self.fioux = nn.Linear(self.in_dim, 4 * self.mem_dim, bias=True) # responsible for bias
        self.iouh = nn.Linear(self.bf * self.mem_dim, self.mem_dim * 3, bias=False)
        self.fh = nn.Linear(self.bf * self.mem_dim, self.bf * self.mem_dim, bias=False)
        
        # self.reg_params = [self.fioux, self.iouh, self.fh]
        self.reg_params = []
    
    def rand_init(self):
        # initialize weight
        for p in [self.fioux, self.iouh, self.fh]:
            utils.init_weight(p.weight)
        
        # initialize forget gate bias
        self.fioux.bias.data.zero_()
        self.fioux.bias.data[:self.mem_dim] = 1
        
    def node_forward(self, inputs, child_c, child_h):
        """
        Args:
            inputs: FloatTensor [in_dim]
            child_c, child_h: FloatTensor [1, bf * mem_dim]
        Returns:
            c, h: FloatTensor [1, mem_dim]
        """
        inputs = inputs.view(1, -1)
        d_inputs = self.forward_dropout(inputs)
        
        fioux = self.fioux(d_inputs) # [1, 4*mem_dim]
        fx, ix, ox, ux = torch.split(fioux, fioux.size(1) // 4, dim=1) # [1, mem_dim]
        iouh = self.iouh(child_h)
        ih, oh, uh = torch.split(iouh, iouh.size(1) // 3, dim=1) # [1, mem_dim]
        i, o, u = F.sigmoid(ix + ih), F.sigmoid(ox + oh), F.tanh(ux + uh)
        
        fh = self.fh(child_h) # [1, bf*mem_dim]
        f = F.sigmoid(fx.repeat(1, self.bf) + fh)
        
        fc = torch.mul(f, child_c).view(-1, self.mem_dim) # [bf, mem_dim]

        c = torch.mul(i, self.semeniuta_dropout(u)) + torch.sum(fc, dim=0, keepdim=True)
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
        
        return [node.state for node in tree]
    
class RelationTreeModel(nn.Module):
    def __init__(self, vocab_size, tagset_size, args):
        super().__init__()
        self.args = args
        self.enable_att = args.attention
        self.use_cell = args.use_cell
        self.dropout_ratio = args.dropout_ratio
        self.embedding_dim = args.embedding_dim
        self.att_dim = args.att_hidden_dim
        self.position = args.position
        self.position_dim = args.position_dim if self.position else 0
        self.position_bound = args.position_bound
        self.position_size = 2*args.position_bound + 1
        self.hidden_dim = args.hidden_dim
        self.tagset_size = tagset_size
        self.word_embeds = nn.Embedding(vocab_size, self.embedding_dim)
        self.relation_embeds = nn.Parameter(torch.Tensor(self.hidden_dim, self.tagset_size)) 
        
        if self.position:
            self.position_embeds = nn.Embedding(self.position_size, self.position_dim)
        
        in_dim = self.embedding_dim + 2*self.position_dim
        
        # build tree model
        if args.childsum_tree:
            self.treernn = ChildSumTreeLSTM(in_dim, self.hidden_dim, dropout=self.dropout_ratio)
        else:
            if args.gru:
                self.treernn = BinaryTreeGRU(in_dim, self.hidden_dim, dropout=self.dropout_ratio)
            else:
                self.treernn = BinaryTreeLSTM(in_dim, self.hidden_dim, dropout=self.dropout_ratio)
        
        # attention
        if self.enable_att:
            # self.attention = IntraAttention(self.hidden_dim)
            self.attention = InterAttention(self.hidden_dim)
            
        self.linear = nn.Linear(self.hidden_dim, self.tagset_size)
        self.dropout1 = nn.Dropout(p=self.dropout_ratio)
        self.dropout2 = nn.Dropout(p=self.dropout_ratio)
        self.reg_params = [self.linear] + self.treernn.reg_params
        if self.enable_att:
            self.reg_params += self.attention.reg_params
      
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
            
        if self.enable_att:
            self.attention.rand_init()
        
        utils.init_embedding(self.relation_embeds)
        # initialize tree
        self.treernn.rand_init()
        
        # initialize linear layer
        utils.init_linear(self.linear)
        
    def update_part_embedding(self, indices):
        hook = utils.update_part_embedding(indices, self.args.cuda)
        self.word_embeds.weight.register_hook(hook)
        
    def forward(self, tree, inputs, pos=None):
        output_dict = {}
        inputs_emb = self.word_embeds(inputs) # [seq_len, w_emb_dim]
        if self.position:
            assert pos is not None
            position_emb = self.position_embeds(pos + self.position_bound) # 2*seq_len, p_embed_dim
            position_emb = torch.cat(torch.split(position_emb, position_emb.size(0) // 2, dim=0), dim=1)
#            position_emb = torch.cat([position_emb[0:position_emb.size(0)//2], position_emb[position_emb.size(0)//2:]], dim=1)
            inputs_emb = torch.cat([inputs_emb, position_emb], dim=1)
            
        states = self.treernn(tree, inputs_emb)
        idx = 0 if self.use_cell else 1 # use cell or hidden states
        hiddens = [state[idx] for state in states]
        
        if self.enable_att:
            hiddens = torch.cat(hiddens, dim=0) # [N, hidden_dim]
            # att_weight = self.attention(hiddens).view(1, -1) # [1, N]
            # att_weight, att_states = self.attention(tree, hiddens)
            # att_weight = att_weight.view(1, -1) # [1, N]
            # att_hiddens = [state[idx] for state in att_states]
            # sent_rep = torch.mm(att_weight, hiddens) # [1, hidden_dim]
            # sent_rep = 0.5 * (hiddens[-1] + att_hiddens[-1])
            att_weight = self.attention(hiddens, self.relation_embeds) # [N, tag_size]
            sent_rep, _ = torch.max(torch.mm(att_weight.transpose(0, 1), hiddens), dim=0, keepdim=True) # [1, hidden_dim]
            output_dict['att_weight'] = att_weight
        else:
            sent_rep = hiddens[-1]
        d_sent_rep = self.dropout2(sent_rep)
        output = self.linear(d_sent_rep) # output: tagset_size
        output_dict['output'] = output
        return output_dict, d_sent_rep
    
    def predict(self, tree, inputs, pos=None):
        output_dict, _ = self.forward(tree, inputs, pos)
        _, pred = torch.max(output_dict['output'].data, dim=1)
        # TODO: check dimensionality
        
        return output_dict, pred