# This file implements F2P SI, namely, F2P flavor that represents only integers, and focuses on improved accuracy on mice.
# The class F2P_si mainly inherits from the class F2P_li. 
# For further details, see "main.tex" in Cntr's Overleaf project.
import math, random, pickle
from printf import printf
import settings
import numpy as np

import F2P_li

class CntrMaster (F2P_li.CntrMaster):
    """
    Generate, check and perform arithmetic operations on F2P counters in SR (Small Reals) flavors.
    The counters are generated as an array with the same format (counterSize and hyperExpSize).
    Then, it's possible to perform arithmetic ops on the counters in chosen indices of the array. 
    """

    flavor = lambda self : 'si'
    
    # Given the vector of the exponent, calculate the value it represents 
    expVec2expVal  = lambda self, expVec, expSize : 2**expSize - 1 + int (expVec, base=2) if expSize>0 else 0    

    def setFlavorParams (self):
        """
        set variables that are unique for 'si' flavor of F2P.
        """
        self.bias           = self.cntrSize - self.hyperSize - 1
        self.expMinVec      = ''
        self.expMinVal      = 0        
        mantMaxSize         = self.cntrSize - self.hyperSize
        self.cntrZeroVec    = '0'*(self.cntrSize)
        self.cntrMaxVec     = '1'*(self.cntrSize)

        self.probOfInc1 = np.empty (self.Vmax)
        for expSize in range(0, self.expMaxSize+1):
            mantSize = self.cntrSize - self.hyperSize - expSize
            for i in range (2**expSize):
                expVec = np.binary_repr(num=i, width=expSize)
                expVal = self.expVec2expVal (expVec=expVec, expSize=expSize)
                resolution = 2**(expVal + self.bias - mantSize)
                self.probOfInc1[abs(expVal)] = 1/resolution
        
        self.probOfInc1[self.Vmax-1] = 1 # Fix the special case, where ExpVal==expMinVal and the resolution is also 1 (namely, prob' of increment is also 1).
        
        # 
        self.cntrppOfAbsExpVal = ['']*(self.Vmax)
        for expSize in range(self.expMaxSize, 0, -1):
            hyperVec = np.binary_repr (expSize, self.hyperSize) 
            mantSize = self.cntrSize - self.hyperSize - expSize
            for i in range (2**expSize-1, 0, -1): 
                expVec = np.binary_repr(num=i, width=expSize)
                expVal = self.expVec2expVal (expVec=expVec, expSize=expSize)
                self.cntrppOfAbsExpVal[abs(expVal)] = hyperVec + np.binary_repr(num=i-1, width=expSize) + '0'*mantSize 
            expVal = self.expVec2expVal (expVec='0'*expSize, expSize=expSize)
            self.cntrppOfAbsExpVal[abs(expVal)] = np.binary_repr (expSize-1, self.hyperSize) + ('1'*(expSize-1) if expSize>1 else '') + '0'*(mantSize+1)

