# This file implements F3P LR, namely, F3P flavor that focuses on improved accuracy on large reals. 
# For further details, see "main.tex" in Cntr's Overleaf project.
import math, random, pickle, numpy as np

import settings, Cntr, F3P_sr
from settings import *
from printf import printf

class CntrMaster (F3P_sr.CntrMaster):
    """
    Generate, check and perform arithmetic operations of the counters. 
    The counters are generated as an array with the same format (counterSize and hyperExpSize).
    Then, it's possible to perform arithmetic ops on the counters in chosen indices of the array. 
    """

    # Given the vector of the exponent, calculate the value it represents 
    expVec2expVal  = lambda self, expVec, expSize : -(2**expSize - 1 + int (expVec, base=2)) if expSize>0 else 0    
    
    flavor = lambda self : 'lr'
    
    def setFlavorParams (self):
        """
        set variables that are unique for 'sr' flavor of F2P.
        """
        self.bias           = 0.5*(self.Vmax-1)
        self.expMinVec      = '1'*self.hyperMaxSize 
        self.expMinVal      = 1 - self.Vmax 
        mantMinSize         = self.cntrSize - 2*self.hyperMaxSize
        self.cntrZeroVec    = '1'*(self.cntrSize-mantMinSize) + '0'*mantMinSize  
        self.cntrMaxVec     = '0' + '1'*(self.cntrSize-1)             
        self.powerMin       = self.expMinVal + self.bias + 1 
   
