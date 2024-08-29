# This file implements F2P SI, namely, F2P flavor that represents only integers, and focuses on improved accuracy on mice.
# The class F2P_si mainly inherits from the class F2P_li. 
# For further details, see "main.tex" in Cntr's Overleaf project.
import math, random, pickle, numpy as np

from printf import printf
import settings, F3P_li
from settings import *

class CntrMaster (F3P_li.CntrMaster):
    """
    Generate, check and perform arithmetic operations on F3P counters in SI flavor.
    The counters are generated as an array with the same format (counterSize and hyperExpSize).
    Then, it's possible to perform arithmetic ops on the counters in chosen indices of the array. 
    """

    flavor = lambda self : 'si'
    
    # Given the vector of the exponent, calculate the value it represents 
    expVec2expVal  = lambda self, expVec, expSize : 2**expSize - 1 + int (expVec, base=2) if expSize>0 else 0    

    def setFlavorParams (self):
        """
        set variables that are unique for 'si' flavor of F3P.
        """
        self.bias           = self.cntrSize - 2
        self.expMinVec      = ''
        self.expMinVal      = 0        
        self.cntrZeroVec    = '0'*(self.cntrSize)
        self.cntrMaxVec     = '1'*(self.cntrSize)

        mantSizeOfHyperSize = [self.cntrSize - 2*hyperSize - 1 for hyperSize in range (self.hyperMaxSize+1)]
        mantSizeOfHyperSize[self.hyperMaxSize] = self.cntrSize - 2*self.hyperMaxSize # for this concrete case, there's no delimiter bit.

        self.probOfInc1 = np.zeros (self.Vmax)
        for hyperSize in range(0, self.hyperMaxSize+1):
            for i in range (2**hyperSize):
                expVec = np.binary_repr(num=i, width=hyperSize) if hyperSize>0 else ''
                expVal = self.expVec2expVal (expVec=expVec, expSize=hyperSize)
                resolution = 2**(expVal + self.bias - mantSizeOfHyperSize[hyperSize])
                self.probOfInc1[expVal] = 1/resolution
        
        self.probOfInc1[0] = 1 # Fix the special case, where ExpVal==expMinVal and the resolution is also 1 (namely, prob' of increment is also 1).
        
        if VERBOSE_DEBUG in self.verbose:
            debugFile = open (f'../res/{self.genSettingsStr()}.txt', 'w')
            printf (debugFile, '// resolutions=\n')
            for item in self.probOfInc1:
                printf (debugFile, '{:.1f}\n' .format (1/item))

        # self.cntrppOfAbsExpVal[e] will hold the next cntr when the (mantissa of the) counter with expVal=e is saturated.
        self.cntrppOfAbsExpVal = [None]*(self.Vmax-1) 
        expVal = 0
        for hyperSize in range(0, self.hyperMaxSize+1):
            for i in range (2**hyperSize-1): 
                expVec = np.binary_repr(num=i+1, width=hyperSize)
                if hyperSize==self.hyperMaxSize: # No delimiter
                    self.cntrppOfAbsExpVal[expVal] = '1'*hyperSize       + expVec + '0'*mantSizeOfHyperSize[hyperSize]
                else:
                    self.cntrppOfAbsExpVal[expVal] = '1'*hyperSize + '0' + expVec + '0'*mantSizeOfHyperSize[hyperSize]
                expVal += 1
            if hyperSize<self.expMaxSize:
                if hyperSize<self.expMaxSize-1:
                    self.cntrppOfAbsExpVal[expVal] = '1'*(hyperSize+1) + '0'*(2 + hyperSize + mantSizeOfHyperSize[hyperSize+1])
                else:
                    self.cntrppOfAbsExpVal[expVal] = '1'*(hyperSize+1) + '0'*(1 + hyperSize + mantSizeOfHyperSize[hyperSize+1])
                expVal += 1

