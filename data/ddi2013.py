#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jyao
"""

import os
import glob
import xml.etree.ElementTree as ET
from tqdm import tqdm
from queue import Queue
from copy import deepcopy
from collections import namedtuple

import nltk

import torch
from torch.utils.data import Dataset

from model.tree import Tree

target_map = {t:i for i, t in enumerate(['null', 'advise', 'effect', 'mechanism', 'int'])}

# p1, p2 are relative positions w.r.t. two drug entity mentions
SingleLine = namedtuple('SingleLine', 'sent_id pair_id e1 e2 ddi type sent p1 p2')

def encode_int_list(l):
    """
    encode integer list to string with space as delimiter
    """
    return ' '.join([str(i) for i in l])

class Word:
    def __init__(self, index, text, etype):
        self.index = index
        self.text = text
        self.etype = etype
    
    def __repr__(self):
        return '[index: {}, text: {}, etype: {}]'.format(self.index, self.text, self.etype)
    
    __str__ = __repr__
        

def parse_charoffset(charoffset):
    """
    Parse charoffset to a tuple containing start and end indices.
    Example:
        charoffset = '3-7;8-9'
        
        [[3, 7], [8, 9]]
    """
    # try split by ';'
    charoffsets = charoffset.split(';')
    return [[int(x.strip()) for x in offset.split('-')] for offset in charoffsets]
    """
    if ';' in charoffset:
        print('Invalid char offset: {}'.format(charoffset))
        return []
    else:
        return [int(x.strip()) for x in charoffset.split('-')]
    """
    
def parse_sentence(sent):
    """
    Parse sentence to get a list of Word class
    Example:
        sent = 'it is it'
        print(parse_sentence(sent))
        
        [(0, 'it', 'O'), (3, 'is', 'O'), (6, 'it', 'O')]
    """
    sent = sent.strip()
    res = []
    if len(sent) == 0:
        return res
    i = j = 0
    while j <= len(sent):
        if j < len(sent) and sent[j] != ' ':
            j += 1
        else:
            if j > i: # in case where two spaces are adjacent
                res.append(Word(i, sent[i:j], 'null'))
            i = j + 1
            j = i
    return res

def tag_word(words, charOffset, eid):
    """
    Args:
        entity: dict has keys charOffset, type, text
    Tag Word with entity type in-place
    Example:
        words = [(0, 'it(', 'O'), (4, 'is', 'O'), (7, 'it', 'O')]
        entity = {'charOffset': [0, 1], 'type': 'eng'} # [inclusive, exclusive]
        print(tag_word(words, entity))
        
        [(0, 'it', 'B-ENG'), (2, (, 'O'), (4, 'is', 'O'), (7, 'it', 'O')]
    """
    beg, end = charOffset
    end += 1
    res = []
    for i, word in enumerate(words):
        if word.index < end and word.index + len(word.text) - 1 >= beg:
            # if there is overlap between char offset and this word
            # tag word
            len_word = len(word.text)
            if word.index < beg:
                head = Word(word.index, word.text[:beg-word.index], 'null')
                res.append(head)
            mention = Word(beg, word.text[beg-word.index:min(len_word, end-word.index)], eid)
            res.append(mention)
            if word.index + len_word > end:
                tail = Word(end, word.text[end-word.index:len_word], 'null')
                res.append(tail)
        else:
            res.append(word)
    return res

def generate_sentences_per_doc(root, position=False):
    """
    Args:
        root: root Element of XML
    """
    for sent_elem in root.findall('sentence'):
        eids = []
        words = parse_sentence(sent_elem.get('text'))
        # tag words with entity id
        for entity in sent_elem.findall('entity'):
            attributes = entity.attrib
            eids.append(attributes['id'])
            charOffset = attributes['charOffset']
            parsed_charoffsets = parse_charoffset(charOffset)
            # in some cases, charOffset is in form of xx-xx;xx-xx, we simply take the first part
            words = tag_word(words, parsed_charoffsets[0], attributes['id'])
            if len(parsed_charoffsets) > 1:
                segment = []
                for charoffset in parsed_charoffsets:
                    segment.append(sent_elem.get('text')[charoffset[0]:charoffset[1]+1])
                print('---------------------')
                print(sent_elem.get('text'))
                print(' '.join(segment))
        
        # replace mention with id        
        sent = []
        for word in words:
            if word.etype == 'null':
                sent.append(word.text)
            elif not sent or word.etype != sent[-1]: # replace consecutive terms into a single one
                sent.append(word.etype)
        sent = ' '.join(sent)
        # for each pair of mentions, generate a sentence
        for pair in sent_elem.findall('pair'):
            sent_blind = sent
            attributes = pair.attrib
            # entity blinding
            e1 = attributes['e1']
            e2 = attributes['e2']
            try:
                etype = 'null' if attributes['ddi'] == 'false' else attributes['type']
            except KeyError:
                print('ddi is true but no type is provided')
                continue
            for eid in eids:
                if eid == e1:
                    new = 'DRUG1'
                elif eid == e2:
                    new = 'DRUG2'
                else:
                    new = 'DRUG0'
                sent_blind = sent_blind.replace(eid, new, 1)
            # tokenize
            sent_blind = nltk.word_tokenize(sent_blind)
            # remove last .
            if sent_blind[-1] == '.':
                sent_blind.pop()
            # ensure there is a pair of mentions
            if 'DRUG1' not in sent_blind or 'DRUG2' not in sent_blind:
                continue
            p1 = ''
            p2 = ''
            if position:
                try:
                    _p1 = sent_blind.index('DRUG1')
                    _p2 = sent_blind.index('DRUG2')
                except:
                    print(sent)
                    print(words)
                    print(e1, e2)
                    raise ValueError
                len_sent = len(sent_blind)
                p1 = [i - _p1 for i in range(0, len_sent)]
                p2 = [i - _p2 for i in range(0, len_sent)]
            yield SingleLine(sent_elem.get('id'), attributes['id'], e1, e2, attributes['ddi'], etype, sent_blind, p1, p2)
    
def write_to_file(data, output_path='../../data/re/drugddi2013/preprocessed/train.ddi'):
    """
    Args: data is list of SingleLine
    """
    with open(output_path, 'w') as fo:
        for single_line in data:
            sent = single_line.sent
            single_line = single_line._replace(sent=' '.join(sent), p1=encode_int_list(single_line.p1), p2=encode_int_list(single_line.p2))
            fo.write('|'.join(single_line))
            fo.write('\n')

def preprocess_ddi(data_path='../../data/re/drugddi2013/raw/train', position=False):
    """
    Preprocess ddi data as follows:
    For each document
        For each sentence in the document
            For each pair in the sentence
                Construct the following line: sent_id|pair_id|e1|e2|ddi|type|sent|p1|p2

    Return:
        res: list of tuples
    """
    
    res = []
    file_pattern = os.path.join(data_path, '*.xml')
    for f in glob.glob(file_pattern):
        print('Processing: {}...'.format(f))
        # import xml data into ElementTree
        tree = ET.parse(f)
        root = tree.getroot()
        for single_line in generate_sentences_per_doc(root, position=position):
            res.append(single_line)
            sent = single_line.sent
            single_line = single_line._replace(sent=' '.join(sent), p1=encode_int_list(single_line.p1), p2=encode_int_list(single_line.p2))
    print('Done')
    return res

def levelOrder(root):
    """
    Binary tree level order traversal
    Args:
        root: Tree
    Return:
        [Tree]
    """
    if root is None:
        return []
    
    q = Queue()
    q.put(root)
    ret = []
    n = 1
    
    while not q.empty():
        count = 0
        for _ in range(n):
            node = q.get()
            ret.append(node)
            for c in node.children:
                if c is not None:
                    q.put(c)
                    count += 1
        n = count
    
    ret.reverse()
    return ret

class DDI2013SeqDataset(Dataset):
    """
    Dataset Class for relation extraction, and sequence model

    args: 
        data_tensor (ins_num, seq_length): words 
        target_tensor (ins_num, 1): targets
    """
    
    def __init__(self, data_tensor, target_tensor, position_tensor, indices, mask):
        assert data_tensor.size(0) == target_tensor.size(0)
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
        self.position_tensor = position_tensor
        self.indices = indices
        self.mask = mask

    def __getitem__(self, index):
        return {'feature': self.data_tensor[index], 
                'position': self.position_tensor[index], 
                'target': self.target_tensor[index], 
                'index': self.indices[index],
                'mask': self.mask[index]}

    def __len__(self):
        return self.data_tensor.size(0)
    
    def collate(self, samples):
        """
        merges a list of samples to form a mini-batch
        """
        def merge(key):
            return torch.stack([s[key] for s in samples], dim=0)
        
        keys = ['feature', 'position', 'target', 'index', 'mask']
        res = {k:merge(k) for k in keys}
        
        return res

class DDI2013TreeDataset(Dataset):
    def __init__(self, path, mapping, caseless, dep=True):
        super().__init__()
        
        self.mapping = mapping
        self.caseless = caseless
        self.dep = dep
        _const = '' if dep else 'c'
        sentences = self.read_sentences(os.path.join(path, 'sent.' + _const + 'toks'))
        self.sentences = self.build_features(sentences)
        self.trees = self.read_trees(os.path.join(path, 'sent.' + _const + 'parents'))
        self.labels = self.read_labels(os.path.join(path, 'other.txt'))
        positions = self.build_positions(sentences)
        self.positions = self.build_features(positions)
        self.size = self.labels.size(0)
        
        # sanity check
        self._sanity_check(os.path.join(path, 'sent.' + _const + 'toks'))

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        tree = deepcopy(self.trees[index])
        sentences = self.sentences[index]
        position = self.positions[index]
        label = self.labels[index]
        return {'feature': sentences, 
                'position': position,
                'tree': tree,
                'target': label}
    
    def collate(self, samples):
        """
        merges a list of samples to form a mini-batch
        Args:
            samples: list of dict
        """
        def aggregate(k):
            return [sample[k] for sample in samples]
        
        assert len(samples) > 0, 'samples contain no element'
        
        keys = samples[0].keys()
        
        res = {k:aggregate(k) for k in keys}
        res['target'] = torch.stack(res['target'], dim=0)
        return res
    
    def build_features(self, inputs):
        return [torch.LongTensor(item) for item in inputs]
    
    def read_sentences(self, filename):
        """
        Args:
            sentences: [[int]]
        """
        with open(filename, 'r') as f:
            sentences = [self.read_sentence(line) for line in tqdm(f.readlines())]
        return sentences

    def read_sentence(self, line):
        def encode(w, mapping, caseless):
            if caseless:
                w = w.lower()
            return mapping.get(w, mapping['UNK'])
        indices = [encode(w, self.mapping, self.caseless) for w in line.strip().split()]
        return indices

    def build_positions(self, sentences):
        """
        Return:
            [[int]], concatenated position indices
        """
        d1, d2 = self.mapping['drug1'], self.mapping['drug2']
        positions = []
        # ensure there is a pair of mentions
        for sent in sentences:
            p1 = None
            p2 = None
            try:
                _p1 = sent.index(d1)
                _p2 = sent.index(d2)
            except:
                print(sent)
                raise ValueError
            len_sent = len(sent)
            p1 = [i - _p1 for i in range(0, len_sent)]
            p2 = [i - _p2 for i in range(0, len_sent)]
            positions.append(p1 + p2)
        return positions
    
    def _sanity_check(self, filename):
        """
        Check if length is consistent for one sentence and corresponding position
        """
        with open(filename, 'r') as f:
            for line, pos in zip(f.readlines(), self.positions):
                sent = list(map(lambda x: str(x), line.split()))
                pos = list(map(lambda x: str(x), pos))
                if len(sent) != len(pos) // 2:
                    print('Inconsistent length {} vs {}! sent: {}\npos: {}'.format(len(sent), len(pos) // 2, ' '.join(sent), ' '.join(pos)))
        
    def read_trees(self, filename):
        with open(filename, 'r') as f:
            trees = [self.read_tree(line) for line in tqdm(f.readlines())]
        return trees

    def read_tree(self, line):
        """
        Parse tree and return level order traversal of tree
        Return:
            [Tree]
        """
        parents = list(map(int, line.split()))
        # sanity check
        assert max(parents) <= len(parents), 'Index should be smaller than length! {}'.format(' '.join(parents))
        trees = dict()
        root = None
        for i in range(1, len(parents) + 1):
            if i - 1 not in trees.keys() and parents[i - 1] != -1:
                idx = i
                prev = None
                while True:
                    parent = parents[idx - 1]
                    if parent == -1:
                        break
                    tree = Tree()
                    if prev is not None:
                        tree.add_child(prev)
                    trees[idx - 1] = tree
                    tree.idx = idx - 1
                    if parent - 1 in trees.keys():
                        trees[parent - 1].add_child(tree)
                        break
                    elif parent == 0:
                        root = tree
                        break
                    else:
                        prev = tree
                        idx = parent

        levelTraversal = levelOrder(root)
        return levelTraversal
    
    def read_labels(self, filename):
        """
        sent_id pair_id e1 e2 ddi type p1 p2
        """
        with open(filename, 'r') as f:
            labels = list(map(lambda x: [target_map[x.split('|')[5]]], f.readlines()))
            labels = torch.LongTensor(labels)
        return labels
