# tree object from stanfordnlp/treelstm

"""
Tree object
Note: is not designed for dynamic construction of tree
"""
class Tree(object):
    __slots__ = ['parent', 'children', 'state', 'idx', 'val']
    def __init__(self):
        self.parent = None
        self.children = []
        self.state = None
        self.idx = None
        self.val = -1

    def add_child(self, child):
        child.parent = self
        self.children.append(child)

    def size(self):
        count = 1
        for c in self.children:
            count += c.size()
        
        return count
    
    def depth(self):
        count = 0
        if self.num_children > 0:
            for c in self.children:
                child_depth = c.depth
                if child_depth > count:
                    count = child_depth()
            count += 1
        
        return count
    
    @property
    def num_children(self):
        return len(self.children)
    
    def __repr__(self):
        def _update_node_idx(node):

            if node.idx is None:
                ret = []
                for c in node.children:
                    ret.append(_update_node_idx(c))

                node.idx = ' '.join(ret)

            return node.idx

        def _repr_node(node, level, indent='\t'):
            ret = []
            ret.append('{} ({})'.format(indent * level + str(node.idx), str(node.val)))
            for c in node.children:
                ret.extend(_repr_node(c, level + 1))

            return ret
        
        _update_node_idx(self)
        ret = _repr_node(self, 0)
        
        return '\n'.join(ret)

'''
class BinaryTree:
    def __init__(self):
        self.parent = None
        self.left = None
        self.right = None
    
    def size(self):
        if getattr(self, '_size'):
            return self._size
        count = 1 + self.left.size() + self.right.size()
        self._size = count
        return self._size
    
    def depth(self):
        if getattr(self, '_depth'):
            return self._depth
        count = 0
        if self.left is not None or self.right is not None:
            child_depth = max(self.left.depth() + self.right.depth())
            if child_depth > count:
                count = child_depth
            count += 1
        self._depth = count
        return self._depth
'''