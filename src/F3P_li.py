# This file implements F3P LI, namely, F3P flavor that represents only integers, and focuses on improved accuracy on elephants.
# The class F3P_li mainly inherits from the class F3P_lr. 
# For further details, see "main.tex" in Cntr's Overleaf project.
import math, random, pickle, numpy as np

from printf import printf
import settings, F3P_lr
from settings import *

class CntrMaster (F3P_lr.CntrMaster):
    """
    Generate, check and perform arithmetic operations on F3P counters in LI flavors.
    The counters are generated as an array with the same format (counterSize and hyperExpSize).
    Then, it's possible to perform arithmetic ops on the counters in chosen indices of the array. 
    """

    flavor = lambda self : 'li'
    
    # Given the vector of the exponent, calculate the value it represents 
    expVec2expVal  = lambda self, expVec, expSize : -(2**expSize - 1 + int (expVec, base=2)) if expSize>0 else 0    

    def setFlavorParams (self):
        """
        set variables that are unique for 'li' flavor of F3P.
        """
        self.bias           = self.cntrSize - 2*self.hyperMaxSize + self.Vmax - 2
        self.expMinVec      = '1'*self.hyperMaxSize
        self.expMinVal      = 1 - self.Vmax        
        mantMinSize         = self.cntrSize - 2*self.hyperMaxSize
        self.cntrZeroVec    = '1'*(self.cntrSize-mantMinSize) + '0'*mantMinSize    
        self.cntrMaxVec     = '0' + '1'*(self.cntrSize-1) 

        self.probOfInc1 = np.zeros (self.Vmax)

        # mantSizeOfHyperSize[h] will hold the mantissa size when the hyperSize is h
        mantSizeOfHyperSize = [self.cntrSize - 2*hyperSize - 1 for hyperSize in range (self.hyperMaxSize+1)]
        mantSizeOfHyperSize[self.hyperMaxSize] = self.cntrSize - 2*self.hyperMaxSize # for this concrete case, there's no delimiter bit.

        for hyperSize in range(0, self.hyperMaxSize+1):
            for i in range (2**hyperSize):
                expVec = np.binary_repr(num=i, width=hyperSize) if hyperSize>0 else ''
                expVal = self.expVec2expVal (expVec=expVec, expSize=hyperSize)
                resolution = 2**(expVal + self.bias - mantSizeOfHyperSize[hyperSize])
                self.probOfInc1[abs(expVal)] = 1/resolution
        
        self.probOfInc1[self.Vmax-1] = 1 # Fix the special case, where ExpVal==expMinVal and the resolution is also 1 (namely, prob' of increment is also 1).
        
        if VERBOSE_DEBUG in self.verbose:
            debugFile = open (f'../res/{self.genSettingsStr()}.txt', 'w')
            printf (debugFile, '// resolutions=\n')
            for item in self.probOfInc1:
                printf (debugFile, '{:.1f}\n' .format (1/item))
            
        if any ([item>1 for item in self.probOfInc1]):
            error (f'F3P_li got entry>1 for self.probOfInc1. self.probOfInc1={self.probOfInc1}')

        # self.cntrppOfAbsExpVal[e] will hold the next cntr when the (mantissa of the) counter with expVal=e is saturated.
        self.cntrppOfAbsExpVal = [None]*self.Vmax 
        absExpVal = self.Vmax-1
        for hyperSize in range(self.hyperMaxSize, 0, -1):
            for i in range (2**hyperSize-1, 0, -1): 
                expVec = np.binary_repr(num=i-1, width=hyperSize)
                if hyperSize==self.hyperMaxSize:
                    self.cntrppOfAbsExpVal[absExpVal] = '1'*hyperSize       + expVec + '0'*mantSizeOfHyperSize[hyperSize] 
                else:
                    self.cntrppOfAbsExpVal[absExpVal] = '1'*hyperSize + '0' + expVec + '0'*mantSizeOfHyperSize[hyperSize]
                absExpVal -= 1 
            if hyperSize>0:
                self.cntrppOfAbsExpVal[absExpVal] = '1'*(hyperSize-1) + '0' + '1'*(hyperSize-1) + '0'*mantSizeOfHyperSize[hyperSize-1]
                absExpVal -= 1
        
    def incCntr (self, cntrIdx=0, factor=int(1), mult=False, verbose=[]):
        """
        Increment the counter to the closest higher value.
        If the cntr reached its max val, or the randomization decides not to inc, merely return the cur cntr.
        Return the counter's value after the increment        
        """
        if factor==1 and mult==False:
            return self.incCntrBy1GetVal (cntrIdx=cntrIdx)
        settings.error ('In F3P_li.incCntr(). Sorry, incCntr is currently supported only for factor=1 and mult=False')
    
    def incCntrBy1GetVal (self, 
            cntrIdx  = 0, # idx of the concrete counter to increment in the array
            forceInc = False # If forceInc==True, increment the counter. Else, inc the counter w.p. corresponding to the next counted value.
        ): 
        """
        Increment the counter to the closest higher value.
        If the cntr reached its max val, or the randomization decides not to inc, merely return the cur cntr.
        Return the counter's value after the increment        
        """
        
        cntr        = self.cntrs[cntrIdx]
        hyperSize   = settings.idxOfLeftmostZero (ar=cntr, maxIdx=self.hyperMaxSize)
        if hyperSize==self.hyperMaxSize: # need no delimiter
            expVec = cntr[hyperSize:2*hyperSize]
        else:
            expVec = cntr[(hyperSize+1):(2*hyperSize+1)]
        expVal     = int (self.expVec2expVal(expVec=expVec, expSize=hyperSize))
        mantSize   = self.cntrSize - 2*hyperSize
        if hyperSize<self.hyperMaxSize:
            mantSize -=1 # recall 1 more bit for the delimiter. 
        mantVec    = cntr[-mantSize:]
        mantIntVal = int (mantVec, base=2)
        mantVal    = float (mantIntVal) / 2**mantSize  

        if expVec == self.expMinVec:
            cntrCurVal = mantVal * (2**self.powerMin)
        else:
            cntrCurVal = (1 + mantVal) * (2**(expVal+self.bias))
        
        if not(forceInc): # check first the case where we don't have to inc the counter 
            if (self.cntrs[cntrIdx]==self.cntrMaxVec or random.random() > self.probOfInc1[abs(expVal)]): 
                return int(cntrCurVal)    

        # now we know that we have to inc. the cntr
        cntrppVal  = cntrCurVal + (1/self.probOfInc1[abs(expVal)])
        if settings.VERBOSE_COUT_CNTRLINE in self.verbose:
            print (f'b4 inc: cntrVec={cntr}, cntrVal={int(cntrCurVal)}')
        if mantVec == '1'*mantSize: # the mantissa overflowed
            self.cntrs[cntrIdx] = self.cntrppOfAbsExpVal[abs(expVal)]
        else:
            if hyperSize<self.hyperMaxSize:
                self.cntrs[cntrIdx] = '1'*hyperSize + '0' + expVec + np.binary_repr(num=mantIntVal+1, width=mantSize) 
            else:
                self.cntrs[cntrIdx] = '1'*hyperSize       + expVec + np.binary_repr(num=mantIntVal+1, width=mantSize) 
        if settings.VERBOSE_COUT_CNTRLINE in self.verbose:
            print (f'after inc: cntrVec={self.cntrs[cntrIdx]}, cntrVal={int(cntrppVal)}')
        return int(cntrppVal) 
        