#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 14:15:07 2017

@author: jyao
"""

from regularizer import Regularizer
import main
if __name__ == '__main__':
    print(Regularizer.PARAMETERS)
    p = 3
    Regularizer.register(p)
    print(Regularizer.PARAMETERS)
    main.test(5)
    print(Regularizer.PARAMETERS)