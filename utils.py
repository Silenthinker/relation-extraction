#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import re
import json
import glob
import argparse
import xml.etree.ElementTree as ET
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.init

from itertools import chain
from collections import Counter
from operator import itemgetter

import nltk

import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import precision_score, recall_score, f1_score

from data import REDataset

''' Example
<document id="DrugDDI.d89" origId="Aciclovir">
	<sentence id="DrugDDI.d89.s0" origId="s0" text="Co-administration of probenecid with acyclovir has been shown to increase the mean half-life and the area under the concentration-time curve.">
		<entity id="DrugDDI.d89.s0.e0" origId="s0.p1" charOffset="21-31" type="drug" text="probenecid"/>
		<entity id="DrugDDI.d89.s0.e1" origId="s0.p2" charOffset="37-47" type="drug" text="acyclovir"/>
		<pair id="DrugDDI.d89.s0.p0" e1="DrugDDI.d89.s0.e0" e2="DrugDDI.d89.s0.e1" interaction="true"/>
	</sentence><sentence id="DrugDDI.d89.s1" origId="s1" text="Urinary excretion and renal clearance were correspondingly reduced."/>
	<sentence id="DrugDDI.d89.s2" origId="s2" text="The clinical effects of this combination have not been studied."/>
</document>
'''

## TODO: tokenization and remove punctuations which are not part of entity mentions

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
		words = [(0, 'it(', O), (4, 'is', O), (7, 'it', O)]
		entity = {'charOffset': [0, 2], 'type': 'eng'} # [inclusive, exclusive]
		print(tag_word(words, entity))
		
		[(0, 'it', 'B-ENG'), (3, (, 'O'), (4, 'is', 'O'), (7, 'it', 'O')]
	"""
	beg, end = charOffset
	end += 1
	origword = None
	orig_i = None
	for i, word in enumerate(words):
		if word.index >= beg and word.index < end: # coarse tagging
			if word.index + len(word.text) - 1 >= end:
				origword = word
				orig_i = i
			word.etype = eid
	# fine tagging
	# if end index of word is larger than end index of charOffset, such as the case of example
	# split the word into two words, and tag the latter O
	if origword is None:
		return
	origtext = origword.text 
	origword.text = origtext[:end - origword.index] # update text
	nextindex = origword.index + len(origword.text)
	nextword = Word(nextindex, origtext[len(origword.text)], 'null')
	words.insert(orig_i + 1, nextword)

def generate_sentences_per_doc(root, tokenize=True):
	"""
	Args:
		root: root Element of XML
	"""
	for sent_elem in root.findall('sentence'):
		eids = []
		words = parse_sentence(sent_elem.get('text'))
		for entity in sent_elem.findall('entity'):
			attributes = entity.attrib
			eids.append(attributes['id'])
			parsed_charoffsets = parse_charoffset(attributes['charOffset'])
			for parsed_charoffset in parsed_charoffsets:
				tag_word(words, parsed_charoffset, attributes['id'])
		sent = []
		for word in words:
			if word.etype == 'null':
				sent.append(word.text)
			elif sent and word.etype != sent[-1]: # replace consecutive terms into a single one
				sent.append(word.etype)
		sent = ' '.join(sent)

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
			if tokenize:
				sent_blind = nltk.word_tokenize(sent_blind)
			yield (sent_elem.get('id'), attributes['id'], e1, e2, attributes['ddi'], etype, sent_blind)

def preprocess_ddi(data_path='../../data/drugddi2013/re/train', output_path='../../data/drugddi2013/re/train.ddi'):
	"""
	Preprocess ddi data as follows:
	For each document
		For each sentence in the document
			For each pair in the sentence
				Construct the following line: sent_id|pair_id|e1|e2|ddi|type|sent

	Return:
		res: list of tuples
	"""
	res = []
	file_pattern = os.path.join(data_path, '*.xml')
	with open(output_path, 'w') as fo:
		for f in glob.glob(file_pattern):
			print('Processing: {}...'.format(f))
			# import xml data into ElementTree
			tree = ET.parse(f)
			root = tree.getroot()
			for sent in generate_sentences_per_doc(root):
				res.append(sent)
				sent = list(sent)
				sent[-1] = ' '.join(sent[-1])
				fo.write('|'.join(sent))
				fo.write('\n')
	print('Done')
	return res

def build_vocab(sents, mini_count=5, caseless=True):
	"""
	Args:
		sents: list of list of strings, [[str]]
		mini_count: threshold to replace rare words with UNK
	"""
	counter = Counter()
	if caseless:
		counter.update([w.lower() for w in chain.from_iterable(sents)])
	else:
		counter.update(chain.from_iterable(sents))
	str2int = [w for w, c in counter.items() if c >= mini_count]
	str2int = {w:i for i, w in enumerate(str2int, 1)} # start from 1
	str2int['UNK'] = 0
	str2int['PAD'] = len(str2int)
	return str2int

def build_features(raw_features, mapping, caseless=True):
	"""
	Args:
		raw_features: list of list of strings
		mappling: dict mapping from word to index
	Return:
		[[int]]
	"""
	def encode(w, mapping, caseless):
		if caseless:
			w = w.lower()
		return mapping.get(w, mapping['UNK'])
	return [list(map(lambda w:encode(w, mapping, caseless), sent)) for sent in raw_features]

def build_targets(raw_targets, mapping):
	"""
	Args:
		raw_targets: [str]
		mapping: dict mappping from string to index
	"""
	return [mapping[target] for target in raw_targets]

def build_corpus(raw_corpus, feature_mapping, target_mapping, caseless):
	"""
	build features and targets
	Args:
		raw_corpus: [([str], str)], list of tuple(features, target)
	"""
	raw_features = [tup[0] for tup in raw_corpus]
	raw_targets = [tup[1] for tup in raw_corpus]
	features = build_features(raw_features, feature_mapping, caseless)
	targets = build_targets(raw_targets, target_mapping)
	return features, targets

def stratified_shuffle_split(features, targets, train_size=0.9):
	"""
	Args:
		inputs: [[int]]
		targets: [int]
	Return:
		(train_features, train_targets), (val_features, val_targets)
	"""
	def np2list(array):
		"""
		numpy array to list
		"""
		return array.tolist()
	X = np.array(features)
	y = np.array(targets)	
	sss = StratifiedShuffleSplit(n_splits=1, train_size=train_size, random_state=0)
	for train_index, val_index in sss.split(X, y):
		train_features, train_targets = X[train_index], y[train_index]
		val_features, val_targets = X[val_index], y[val_index]
		break
	return np2list(train_features), np2list(train_targets), np2list(val_features), np2list(val_targets)

def calc_threshold_mean(features):
    """
    calculate the threshold for bucket by mean
    """
    lines_len = list(map(lambda t: len(t) + 1, features))
    average = int(sum(lines_len) / len(lines_len))
    lower_line = list(filter(lambda t: t < average, lines_len))
    upper_line = list(filter(lambda t: t > average, lines_len))
    if lower_line:
    	lower_average = int(sum(lower_line) / len(lower_line))
    else:
    	lower_average = None
    if upper_line:
    	upper_average = int(sum(upper_line) / len(upper_line))
    else:
    	upper_average = None
    max_len = max(lines_len)
    max_len = None if max_len == upper_average else max_len
    return [line_len for line_len in [lower_average, average, upper_average, max_len] if line_len]

def construct_bucket_dataloader(input_features, input_targets, pad_feature, caseless, batch_size, is_train=True):
    """
    Construct bucket
    """
    # encode and padding
    thresholds = calc_threshold_mean(input_features)
    buckets = [[[], []] for _ in range(len(thresholds))]
    for feature, target in zip(input_features, input_targets):
        cur_len = len(feature)
        idx = 0
        cur_len_1 = cur_len + 1
        while thresholds[idx] < cur_len_1:
            idx += 1
        buckets[idx][0].append(feature + [pad_feature] * (thresholds[idx] - cur_len))
        buckets[idx][1].append([target])
    bucket_dataset = [REDataset(torch.LongTensor(bucket[0]), torch.LongTensor(bucket[1]))
                      for bucket in buckets]
    dataset_loader = [torch.utils.data.DataLoader(tup, batch_size, shuffle=is_train, drop_last=False) for tup in bucket_dataset]
    return dataset_loader

def init_embedding(input_embedding):
    """
    Initialize embedding
    """
    bias = np.sqrt(3.0 / input_embedding.size(1))
    nn.init.uniform(input_embedding, -bias, bias)

def init_linear(input_linear):
    """
    Initialize linear transformation
    """
    bias = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.weight.size(1)))
    nn.init.uniform(input_linear.weight, -bias, bias)
    if input_linear.bias is not None:
        input_linear.bias.data.zero_()

def init_lstm(input_lstm):
    """
    Initialize lstm
    """
    for ind in range(0, input_lstm.num_layers):
        weight = eval('input_lstm.weight_ih_l'+str(ind))
        bias = np.sqrt(6.0 / (weight.size(0)/4 + weight.size(1)))
        nn.init.uniform(weight, -bias, bias)
        weight = eval('input_lstm.weight_hh_l'+str(ind))
        bias = np.sqrt(6.0 / (weight.size(0)/4 + weight.size(1)))
        nn.init.uniform(weight, -bias, bias)
    
    if input_lstm.bias:
        for ind in range(0, input_lstm.num_layers):
            weight = eval('input_lstm.bias_ih_l'+str(ind))
            weight.data.zero_()
            weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
            weight = eval('input_lstm.bias_hh_l'+str(ind))
            weight.data.zero_()
            weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1

def save_checkpoint(state, track_list, filename):
	"""
	save checkpoint
	"""
	with open(filename + '.json', 'w') as f:
		json.dump(track_list, f)
	torch.save(state, filename + '.model')

def build_parser():
	parser = argparse.ArgumentParser(description='Learning to extract relation')
	# load and save
	parser.add_argument('--train_corpus_path', default='../../data/drugddi2013/re/train', help='path to original train corpus')
	parser.add_argument('--test_corpus_path', default='../../data/drugddi2013/re/test', help='path to original test corpus')
	parser.add_argument('--train_path', default='../../data/drugddi2013/re/train.ddi', help='path to train data')
	parser.add_argument('--test_path', default='../../data/drugddi2013/re/test.ddi', help='path to test data')
	parser.add_argument('--checkpoint', default='./checkpoint/re', help='path to checkpoint prefix')
	parser.add_argument('--load_checkpoint', default='./checkpoint/re_lstm.model', help='path to load checkpoint')

	# preprocess
	parser.add_argument('--train_size', default=0.8, help='split train corpus into train/val set according to the ratio')
	parser.add_argument('--caseless', action='store_true', help='caseless or not')

	# model
	parser.add_argument('--embedding_dim', type=int, default=100, help='embedding dimension')
	parser.add_argument('--hidden_dim', type=int, default=100, help='hidden layer dimension')
	parser.add_argument('--rnn_layers', type=int, default=1, help='number of rnn layers')
	parser.add_argument('--dropout_ratio', type=float, default=0.4, help='dropout ratio')

	# training
	parser.add_argument('--lr', default=0.01, help='learning rate')
	parser.add_argument('--lr_decay', type=float, default=0.001, help='decay ratio of learning rate')
	parser.add_argument('--momentum', default=0.9, help='momentum for sgd')
	parser.add_argument('--clip_grad_norm', default=0.5, help='clip gradient norm')
	parser.add_argument('--batch_size', type=int, default=128, help='batch size')
	parser.add_argument('--num_epoch', type=int, default=200, help='number of epochs')
	parser.add_argument('--patience', type=int, default=15, help='patience for early stop')
	parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
	
	
	return parser

def load_corpus(train_path, test_path):
	"""
	load ddi corpus
	"""
	def load_file(path):
		def parse_line(line):
			line = line.split('|')
			line[-1] = line[-1].split()
			return tuple(line)

		corpus = None
		try:
			with open(path, 'r') as f:
				corpus = [parse_line(line) for line in f]
		except Exception as inst:
			print(inst)
		return corpus	
	return load_file(train_path), load_file(test_path)
	
def evaluate(y_true, y_pred, labels=None):
	"""
	calculate (micro) precision, recall, and f1 score
	Args:
		labels: [int] labels to consider (e.g., skip false ddi relation)
	"""
	average = 'micro'
	def _evaluate(scorer):
		return scorer(y_true, y_pred, labels=labels, average=average)
	precision = _evaluate(precision_score)
	recall = _evaluate(recall_score)
	f1 = _evaluate(f1_score)
	return precision, recall, f1

def adjust_learning_rate(optimizer, lr):
    """
    shrink learning rate for pytorch
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
	# res = preprocess_ddi()
	caseless = True
	batch_size = 2
	raw_corpus = [
		(['This', 'is', 'an', 'example', '.'], 'mechanism'), 
		(['this', 'is', 'also', 'an', 'example', '.'], 'int'),
		(['this', 'is', 'example', '.'], 'int'),
		(['This', 'is', 'an', 'example', '.'], 'mechanism'), 
		(['this'], 'int'),
		]
	sents = [tup[0] for tup in raw_corpus]
	feature_mapping = build_vocab(sents, mini_count=1, caseless=caseless)
	target_mapping = {'null':0, 'advice':1, 'effect':2, 'mechanism':3, 'int':4}
	input_features, input_targets = build_corpus(raw_corpus, feature_mapping, target_mapping, caseless)
	dataset_loader = construct_bucket_dataloader(input_features, input_targets, feature_mapping['PAD'], caseless, batch_size, True)
	for feature, target in chain.from_iterable(dataset_loader):
		print(feature)
		print(target)
		print('-------')



