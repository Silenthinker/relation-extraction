#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 13:50:09 2018

@author: jyao
"""
import os
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def load_json_from_file(filepath):
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except IOError:
        raise

def parse_track_list(tl):
    """
    {"track_list": 
        [
            {"epoch": 0, 
            "loss": 0.5571183884190133, 
            "dev_prec": 0.36363636363636365, 
            "dev_rec": 0.009950248756218905, 
            "dev_f1": 0.019370460048426148, 
            "dev_loss": 0.58050367096439, 
            "test_prec": 0.71875, 
            "test_rec": 0.02349336057201226, 
            "test_f1": 0.04549950544015826
            }, 
        ]
    }
    """
    epochs = []
    test_f1 = []
    
    for item in tl:
        if 'test_f1' in item:
            epoch = item['epoch']
            epochs.append(epoch)
            test_f1.append(item['test_f1'])
            if epoch >= 50:
                break
    return np.array(epochs), np.array(test_f1)
    
if __name__ == '__main__':
    filenames = ('seq_lstm.json', 'dep_tree_lstm.json', 'const_tree_lstm.json')
    filepaths = (os.path.join('checkpoint', fname) for fname in filenames)
    
    # read json files
    train_info = []
    for fpath in filepaths:
        json_obj = load_json_from_file(fpath)
        tl = json_obj['track_list']
        # extract track list
        train_info.append(parse_track_list(tl))
    
    tr1, tr2, tr3 = train_info
    def _last_epoch(tr):
        x , y = tr
        return x[-1]
    
    def _best_score(tr):
        x, y = tr
        return y[-1]
    
    # plot convergence
    fontsize = 22
    lw = 3
    font = {'family': 'normal',
        'size': 18}

    matplotlib.rc('font', **font)
    fig = plt.figure(figsize=(16, 9))
    ax = fig.subplots()
    plt.plot(*tr1, 'b', *tr2, 'g', *tr3, 'C9', lw=lw)
#    plt.axhline(max(map(_best_score, train_info)), color='C1', ls='-.', lw=lw)
    plt.yticks(np.arange(0, 0.75, 0.1))
    ax.set_yticks(np.arange(0, 0.75, 0.025), minor=True)
    ax.grid(which='both')
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)
    plt.legend(['Chain-structured LSTM', 'Dependency Tree-LSTM', 'Constituency Tree-LSTM'], loc='lower right')
    plt.xlabel('Epoch', fontsize=fontsize)
    plt.ylabel('F score', fontsize=fontsize)
    
    plt.savefig('convergence.pdf')