# tree object from stanfordnlp/treelstm

"""
Tree object
Note: is not designed for dynamic construction of tree
"""
class Tree(object):
    __slots__ = ['parent', 'children', 'state', '_size', '_depth', 'idx']
    def __init__(self):
        self.parent = None
        self.children = []
        self.state = None
        self.size = None
        self.depth = None
        self.idx = None

    def add_child(self, child):
        child.parent = self
        self.children.append(child)

    @property
    def size(self):
        if self._size is None:
            count = 1
            for c in self.children:
                count += c.size
            self._size = count
        return self._size
    
    @size.setter
    def size(self, n):
        self._size = n

    @property
    def depth(self):
        if self._depth is None:
            count = 0
            if self.num_children > 0:
                for c in self.children:
                    child_depth = c.depth
                    if child_depth > count:
                        count = child_depth
                count += 1
            self._depth = count
        return self._depth
    
    @depth.setter
    def depth(self, n):
        self._depth = n
    
    @property
    def num_children(self):
        return len(self.children)

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