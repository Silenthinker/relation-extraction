#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 19:09:59 2018

@author: jyao
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


f1_by_len_tree = [(30, 0.77634011090573007), (60, 0.66458658346333854), (100, 0.87096774193548387)]
f1_by_len_seq = [(30, 0.68705882352941172), (60, 0.60805860805860812), (100, 0.41350210970464135)]

f1_by_len_tree = tuple(zip(*f1_by_len_tree))
f1_by_len_seq = tuple(zip(*f1_by_len_seq))
#    lens = (20, 40, 60, 80, 100)
#    f1 = (0.82314049586776861, 0.7431302270011948, 0.52142857142857146, 0.89156626506024106, 0.82926829268292679)

fontsize = 22
font = {'family': 'normal',
    'size': 20}

matplotlib.rc('font', **font)
fig = plt.figure(figsize=(16, 9))
ax = fig.subplots()
plt.plot(*f1_by_len_tree, '^b', *f1_by_len_seq, '+g',lw = 3)
plt.xticks(f1_by_len_tree[0])
plt.yticks(np.arange(0, 1.0, 0.1))
ax.set_yticks(np.arange(0, 1.0, 0.05), minor=True)
ax.grid(which='both')
ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.5)
plt.xlabel('Sentence length', fontsize=fontsize)
plt.ylabel('F score', fontsize=fontsize)
plt.legend(['Dependency Tree-LSTM', 'Chain-structured LSTM'], 'top right')

plt.savefig('f1_by_len.pdf')