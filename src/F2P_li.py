# This file implements F2P LI, namely, F2P flavor that represents only integers, and focuses on improved accuracy on elephants.
# The class F2P_lr mainly inherits from the class F2P_lr. 
# For futher details, see "main.tex" in Cntr's Overleaf project.
import math, random, pickle
from printf import printf
import settings
import numpy as np

import F2P_lr

class CntrMaster (F2P_lr.CntrMaster):
    """
    Generate, check and perform arithmetic operations on F2P counters in SR (Small Reals) flavors.
    The counters are generated as an array with the same format (counterSize and hyperExpSize).
    Then, it's possible to perform arithmetic ops on the counters in chosen indices of the array. 
    """

    flavor = lambda self : 'li'
    
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

        self.probOfIncBy1 = np.empty (self.Vmax)
        for expSize in range(0, self.expMaxSize+1):
            mantSize = self.cntrSize - self.hyperSize - expSize
            for i in range (2**expSize):
                expVec = np.binary_repr(num=i, width=expSize)
                expVal = self.expVec2expVal (expVec=expVec, expSize=expSize)
                resolution = 2**(expVal + self.bias -mantSize)
                # print (f'expSize={expSize}, expVec={expVec}, expVal={expVal}, mantSize={mantSize}, resolution={resolution}, inv(res)={1/resolution}')
                self.probOfIncBy1[abs(expVal)] = 1/resolution
        
        self.probOfIncBy1[self.Vmax-1] = 1 # Fix the special case, where ExpVal==expMinVal and the resolution is also 1 (namely, prob' of increment is also 1).

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
        mantVal    = float (int (mantVec, base=2)) / 2**(self.cntrSize - self.hyperSize - expSize)  
        cntrCurVal = self.cntr2num(cntr)

        if expVec == self.expMinVec:
            cntrCurVal = mantVal * (2**self.powerMin)
        else:
            cntrCurVal = (1 + mantVal) * (2**(expVal+self.bias))
        
        if not(forceInc): # check first the case where we don't have to inc the counter 
            if (self.cntrs[cntrIdx]==self.cntrMaxVec or random.random() > self.probOfInc1[abs(expVal)]): 
                return cntrCurVal    

        # now we know that we have to inc. the cntr
        cntrppVal  = cntrCurVal + (1/self.probOfInc1[abs(expVal)])
        print (f'b4 inc: cntrVec={cntr}, cntrVal={cntrCurVal}')
        if mantVec == '1'*mantSize: # the mantissa overflowed
            settings.error ('TBD')
        else:
            self.cntrs[cntrIdx] = hyperVec + expVec + np.binary_repr(num=mantVal+1, width=mantSize) 
        print (f'after inc: cntrVec={self.cntrs[cntrIdx]}, cntrVal={cntrppVal}')
        return cntrppVal 
        
            