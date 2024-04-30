# This file implements F2P LI, namely, F2P flavor that represents only integers, and focuses on improved accuracy on elephants.
# The class F2P_li mainly inherits from the class F2P_lr. 
# For further details, see "main.tex" in Cntr's Overleaf project.
import math, random, pickle, numpy as np

from printf import printf
import settings, F2P_lr
from settings import VERBOSE_DEBUG

class CntrMaster (F2P_lr.CntrMaster):
    """
    Generate, check and perform arithmetic operations on F2P counters in SR (Small Reals) flavors.
    The counters are generated as an array with the same format (counterSize and hyperExpSize).
    Then, it's possible to perform arithmetic ops on the counters in chosen indices of the array. 
    """

    flavor = lambda self : 'li'
    
    # Given the vector of the exponent, calculate the value it represents 
    expVec2expVal  = lambda self, expVec, expSize : -(2**expSize - 1 + int (expVec, base=2)) if expSize>0 else 0    

    def setFlavorParams (self):
        """
        set variables that are unique for 'li' flavor of F2P.
        """
        self.bias           = self.cntrSize - self.hyperSize - 2**self.hyperSize + self.Vmax - 1
        self.expMinVec      = '1'*(2**self.hyperSize-1)
        self.expMinVal      = -self.Vmax+1        
        mantMinSize         = self.cntrSize - (self.hyperSize + 2**self.hyperSize-1)
        self.cntrZeroVec    = '1'*(self.cntrSize-mantMinSize) + '0'*mantMinSize  #np.binary_repr(0, self.cntrSize)  
        self.cntrMaxVec     = '0'*self.hyperSize + '1'*(self.cntrSize-self.hyperSize) #np.binary_repr((1<<self.cntrSize)-1, self.cntrSize)

        self.probOfInc1 = np.zeros (self.Vmax)
        for expSize in range(0, self.expMaxSize+1):
            mantSize = self.cntrSize - self.hyperSize - expSize
            for i in range (2**expSize):
                expVec = np.binary_repr(num=i, width=expSize)
                expVal = self.expVec2expVal (expVec=expVec, expSize=expSize)
                resolution = 2**(expVal + self.bias -mantSize)
                self.probOfInc1[abs(expVal)] = 1/resolution
        
        self.probOfInc1[self.Vmax-1] = 1 # Fix the special case, where ExpVal==expMinVal and the resolution is also 1 (namely, prob' of increment is also 1).
        
        if VERBOSE_DEBUG in self.verbose:
            debugFile = open (f'../res/{self.genSettingsStr()}.txt', 'w')
            printf (debugFile, '// resolutions=\n')
            for item in self.probOfInc1:
                printf (debugFile, '{:.1f}\n' .format (1/item))
            
        if any ([item>1 for item in self.probOfInc1]):
            error (f'F2P_li Got entry>1 for self.probOfInc1. self.probOfInc1={self.probOfInc1}')
        # self.cntrppOfAbsExpVal[e] will hold the next cntr when the (mantissa of the) counter with expVal=e is saturated.
        self.cntrppOfAbsExpVal = ['' for _ in range(self.Vmax)]
        for expSize in range(self.expMaxSize, 0, -1):
            hyperVec = np.binary_repr (expSize, self.hyperSize) 
            mantSize = self.cntrSize - self.hyperSize - expSize
            for i in range (2**expSize-1, 0, -1): 
                expVec = np.binary_repr(num=i, width=expSize)
                expVal = self.expVec2expVal (expVec=expVec, expSize=expSize)
                self.cntrppOfAbsExpVal[abs(expVal)] = hyperVec + np.binary_repr(num=i-1, width=expSize) + '0'*mantSize 
            expVal = self.expVec2expVal (expVec='0'*expSize, expSize=expSize)
            self.cntrppOfAbsExpVal[abs(expVal)] = np.binary_repr (expSize-1, self.hyperSize) + ('1'*(expSize-1) if expSize>1 else '') + '0'*(mantSize+1)

    def incCntr (self, cntrIdx=0, factor=int(1), mult=False, verbose=[]):
        """
        Increment the counter to the closest higher value.
        If the cntr reached its max val, or the randomization decides not to inc, merely return the cur cntr.
        Return the counter's value after the increment        
        """
        if factor==1 and mult==False:
            return self.incCntrBy1GetVal (cntrIdx=cntrIdx)
        settings.error ('In F2P_li.incCntr(). Sorry, incCntr is currently supported only for factor=1 and mult=False')
    
    def incCntrBy1GetVal (self, 
                    cntrIdx  = 0, # idx of the concrete counter to increment in the array
                    forceInc = False # If forceInc==True, increment the counter. Else, inc the counter w.p. corresponding to the next counted value.
                    ): 
        """
        Increment the counter to the closest higher value.
        If the cntr reached its max val, or the randomization decides not to inc, merely return the cur cntr.
        Return the counter's value after the increment        
        """
        
        cntr       = self.cntrs[cntrIdx]
        hyperVec   = cntr [0:self.hyperSize]
        expSize    = int(hyperVec, base=2)
        expVec     = cntr[self.hyperSize:self.hyperSize+expSize]
        expVal     = int (self.expVec2expVal(expVec, expSize))
        mantSize   = self.cntrSize - self.hyperSize - expSize 
        mantVec    = cntr[-mantSize:]
        mantIntVal = int (mantVec, base=2)
        mantVal    = float (mantIntVal) / 2**(self.cntrSize - self.hyperSize - expSize)  

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
            self.cntrs[cntrIdx] = hyperVec + expVec + np.binary_repr(num=mantIntVal+1, width=mantSize) 
        if settings.VERBOSE_COUT_CNTRLINE in self.verbose:
            print (f'after inc: cntrVec={self.cntrs[cntrIdx]}, cntrVal={int(cntrppVal)}')
        return int(cntrppVal) 
        
            