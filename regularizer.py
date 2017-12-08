#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

class Regularizer:
    """
    wrapper for regularization
    """
    def __init__(self, variables, reg_lambda, reg_type='l2'):
        """
        Args:
            variables: [Variable]
            reg_lambda: float, regularization scale
            reg_type: str, l1, l2
        """
        
        self.__reg_lambda = reg_lambda
        assert reg_type in ['l1', 'l2'], 'regularization type must be either l1 or l2'
        self.__reg_type = reg_type
        self.__reg_vars = variables
        
    @property
    def reg_lambda(self):
        return self.__reg_lambda
    
    @property
    def reg_type(self):
        return self.__reg_type
    
    @property
    def reg_vars(self):
        return self.__reg_vars
    
    def __call__(self):
        """
        compute regularization term = reg_lambda * norm(vars)
        """
        p = 1 if self.reg_type == 'l1' else 2
        reg = None
        for v in self.reg_vars:
            norm = v.norm(p)
            # if norm is not equal 0
            if norm.data[0] > 0:
                if reg is None: # trick to make sure bp of variable∂
                    reg = v.norm(p)
                else:
                    reg = reg + v.norm(p)
                
        return self.reg_lambda * reg
            
    
        
    
    

        
    