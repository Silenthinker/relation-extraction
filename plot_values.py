#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 09:31:25 2018

@author: jyao
"""
import numpy as np
import matplotlib.pyplot as plt

## plot values from file

if __name__ == '__main__':
    filename = 'norm.txt'
    vals = []
    with open(filename, 'r') as f:
        for i in f:
            vals.append(float(i.strip()))
    
    vals = np.array(vals)
    plt.plot(range(len(vals)), vals, 'b')
    plt.xlabel('Batch')
    plt.ylabel('2-Norm')
    plt.title('2-norm of weight matrix in attention')
