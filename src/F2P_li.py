# This file implements F2P LI, namely, F2P flavor that represents only integers, and focuses on improved accuracy on elephants.
# The class F2P_li mainly inherits from the class F2P_lr. 
# For further details, see "main.tex" in Cntr's Overleaf project.
import math, random, pickle, numpy as np

from printf import printf
import settings, F2P_lr
from settings import VERBOSE_DEBUG, VERBOSE_LOG, warning, error

class CntrMaster (F2P_lr.CntrMaster):
    """
    Generate, check and perform arithmetic operations of the counters. 
    The counters are generated as an array with the same format (counterSize and hyperExpSize).
    Then, it's possible to perform arithmetic ops on the counters in chosen indices of the array. 
    """

    flavor = lambda self : 'li'
    
    # Given the vector of the exponent, calculate the value it represents 
    expVec2expVal  = lambda self, expVec, expSize : -(2**expSize - 1 + int (expVec, base=2)) if expSize>0 else 0    

    def halfAllCntrs (self):
        """
        Half the values of all the counters
        """
        mantSizeBase = self.cntrSize - self.hyperSize # default value for the mantissa; to be refined within the loop 
        for num in range(2 ** self.cntrSize): #$$$
            cntr = np.binary_repr(num, self.cntrSize) #$$$
        # for cntr in self.cntrs: #$$$
            hyperVec  = cntr [0:self.hyperSize]
            expSize   = int(hyperVec, base=2)  
            mantSize  = mantSizeBase - expSize  
            expVec    = cntr[self.hyperSize:self.hyperSize+expSize]
            absExpVal = abs(self.expVec2expVal(expVec=expVec, expSize=expSize))
            mantVec   = cntr[self.hyperSize+expSize:] 

            # Need to code the special case of (sub) normal values.
            if VERBOSE_LOG in self.verbose:
                orgVal = self.cntr2num (cntr)
                printf (self.logFile, 'cntr={}, absExpVal={}, orgVal={:.0f} ' .format(cntr, absExpVal, orgVal))
            if absExpVal==self.Vmax-1: # The edge case of sub-normal values: need to only divide the mantissa; no need (and cannot) further decrease the exponent
                if mantVec[-1]=='1':
                    truncated = True
                mantVec = mantVec >> 1 # divide the mantissa by 2 (by right-shift) 
                if truncated and random.random()>=0.5: # if the removed bit was '1', add '1' w.p. 0.5 
                    mantVec = np.binary_repr(num=int(mantVec, base=2)+1, mantSize)    
                cntr = hyperVec + expVec + mantVec
            elif self.mantSizeOfAbsExpVal[absExpVal]==mantSize: 
                cntr = self.LsbVecOfAbsExpVal[absExpVal] + mantVec
            # Now we know that the mantissa field of the halved cntr should be 1-bit shorter
            elif mantVec[-1]=='0' or random.random()<0.5: # the lsb is reset, or we should trunc it 
                cntr = self.LsbVecOfAbsExpVal[absExpVal] + mantVec[0:-1]

            # should ceil the mantissa
            elif mantVec=='1'*mantSize: # The mantissa vector is "11...1" --> should further increase the exponent
                cntr = self.LsbVecOfAbsExpVal[absExpVal+1] + '0'*self.mantSizeOfAbsExpVal[absExpVal+1]
            else:
                mantVal = int (mantVec, base=2)
                cntr = self.LsbVecOfAbsExpVal[absExpVal] + np.binary_repr(mantVal+1, mantSize)[0:-1]
            if VERBOSE_LOG in self.verbose:
                val = self.cntr2num (cntr)
                printf (self.logFile, f'halvedCntr={cntr} ')
                if val==float(orgVal)/2:
                    printf (self.logFile, f'vals fit\n')
                else:
                    printf (self.logFile, 'val={:.0f}\n' .format(val))                    
                
                
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
        
        self.cntrppOfAbsExpVal   = ['']*(self.Vmax) # self.cntrppOfAbsExpVal[e] will hold the next cntr when the (mantissa of the) counter with expVal=e is saturated.
        self.LsbVecOfAbsExpVal   = ['']*(self.Vmax) # self.LsbVecOfAbsExpVal[e] will hold the LSB fields (hyperVec and expVec) of the vector when decreasing the vector's exponent whose current absolute value is e.
        for expSize in range(self.expMaxSize, 0, -1):
            hyperVec = np.binary_repr (expSize, self.hyperSize) 
            mantSize = self.cntrSize - self.hyperSize - expSize
            for i in range (2**expSize-1, 0, -1): 
                expVec = np.binary_repr(num=i, width=expSize)
                expVal = self.expVec2expVal (expVec=expVec, expSize=expSize)
                self.cntrppOfAbsExpVal[abs(expVal)] = hyperVec + np.binary_repr(num=i-1, width=expSize) + '0'*mantSize
                self.LsbVecOfAbsExpVal[abs(expVal)-1] = hyperVec + expVec 
            expVal = self.expVec2expVal (expVec='0'*expSize, expSize=expSize)
            self.cntrppOfAbsExpVal[abs(expVal)] = np.binary_repr (expSize-1, self.hyperSize) + ('1'*(expSize-1) if expSize>1 else '') + '0'*(mantSize+1)
            self.LsbVecOfAbsExpVal[abs(expVal)-1] = hyperVec + '0'*expSize
        self.LsbVecOfAbsExpVal[self.Vmax-1] = '1'*(self.hyperSize + 2**self.hyperSize - 1) 
        self.mantSizeOfAbsExpVal = [self.cntrSize - len(item) for item in self.LsbVecOfAbsExpVal] # self.mantSizeOfAbsExpVal[e] is the size of the mantissa field of the vector when decreasing the vector's exponent whose current absolute value is e.
        
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
  
#$$$      
myF2P_li_cntr = CntrMaster (
    cntrSize    = 6, 
    hyperSize   = 2,
    verbose     = [VERBOSE_LOG]
) 
logFile = open (f'../res/log_files/{myF2P_li_cntr.genSettingsStr()}.log', 'w')
myF2P_li_cntr.setLogFile (logFile)
myF2P_li_cntr.halfAllCntrs()
            