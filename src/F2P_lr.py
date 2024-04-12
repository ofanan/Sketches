# This file implements F2P LR, namely, F2P flavor that focuses on improved accuracy on large reals.
# The class F2P_lr mainly inherits from the class F2P_sr. 
# For futher details, see "main.tex" in Cntr's Overleaf project.
import math, random, pickle
from printf import printf
import settings
import numpy as np

import F2P_sr

class CntrMaster (F2P_sr.CntrMaster):
    """
    Generate, check and perform arithmetic operations on F2P counters in SR (Small Reals) flavors.
    The counters are generated as an array with the same format (counterSize and hyperExpSize).
    Then, it's possible to perform arithmetic ops on the counters in chosen indices of the array. 
    """

    # Given the vector of the exponent, calculate the value it represents 
    expVec2expVal  = lambda self, expVec, expSize : -(2**expSize - 1 + int (expVec, base=2)) if expSize>0 else 0    
    
    flavor = lambda self : 'lr'
    
    # Given the value of the exponent, return the exponent vector representing this value 
    # expVal2expVec       = lambda self, expVal, expSize : np.binary_repr(num=int(self.biasOfExpSize[int(expSize)]) - expVal, width=expSize) if expSize>0 else ""   

    def setFlavorParams (self):
        """
        set variables that are unique for 'lr' flavor of F2P.
        """
        self.bias           = 0.5*(self.Vmax-1)
        self.expMinVec      = '1'*(2**self.hyperSize-1) 
        self.expMinVal      = -self.Vmax+1 
        mantMinSize         = self.cntrSize - (self.hyperSize + 2**self.hyperSize-1)
        self.cntrZeroVec    = '1'*(self.cntrSize-mantMinSize) + '0'*mantMinSize  #np.binary_repr(0, self.cntrSize)  
        self.cntrMaxVec     = '0'*self.hyperSize + '1'*(self.cntrSize-self.hyperSize) #np.binary_repr((1<<self.cntrSize)-1, self.cntrSize)
