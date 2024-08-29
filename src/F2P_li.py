# This file implements F2P LI, namely, F2P flavor that represents only integers, and focuses on improved accuracy on elephants.
# The class F2P_li mainly inherits from the class F2P_lr. 
# For further details, see "main.tex" in Cntr's Overleaf project.
import math, random, pickle, numpy as np

from printf import printf
import settings, F2P_lr
from settings import *

class CntrMaster (F2P_lr.CntrMaster):
    """
    Generate, check and perform arithmetic operations of the counters. 
    The counters are generated as an array with the same format (counterSize and hyperExpSize).
    Then, it's possible to perform arithmetic ops on the counters in chosen indices of the array. 
    """

    flavor = lambda self : 'li'
    
    # Given the vector of the exponent, calculate the value it represents 
    expVec2expVal  = lambda self, expVec, expSize : -(2**expSize - 1 + int (expVec, base=2)) if expSize>0 else 0    

    def upScale (self):
        """
        Allow down-sampling:
        - Half the values of all the counters.
        - Increase the bias value added to the exponent, to return the counters to roughly their original values.
        """
        mantSizeBase = self.cntrSize - self.hyperSize # default value for the mantissa; to be refined within the loop 
        for cntrIdx in range(self.numCntrs):
            cntr      = self.cntrs[cntrIdx]   
            hyperVec  = cntr [0:self.hyperSize]
            expSize   = int(hyperVec, base=2)  
            mantSize  = mantSizeBase - expSize  
            expVec    = cntr[self.hyperSize:self.hyperSize+expSize]
            absExpVal = abs(self.expVec2expVal(expVec=expVec, expSize=expSize))
            mantVec   = cntr[self.hyperSize+expSize:] 

            # Need to code the special case of (sub) normal values.
            if VERBOSE_DEBUG in self.verbose:
                orgVal = self.cntr2num (cntr)
            truncated = False # By default, we didn't truncate the # when dividing by 2 --> no need to round. 
            
            if absExpVal==self.Vmax-1: # The edge case of sub-normal values: need to only divide the mantissa; no need (and cannot) further decrease the exponent
                if mantVec[-1]=='1':
                    truncated = True
                mantVec = '0' + mantVec[0:-1] # mantVec >> 1 # divide the mantissa by 2 (by right-shift) 
            elif absExpVal==self.Vmax-1: # The edge case of 1-above sub-normal values: need to only right-shift the value, and insert '1' in the new leftmost mantissa bit. 
                if mantVec[-1]=='1':
                    truncated = True
                mantVec = '1' + mantVec[0:-1]                
            elif self.mantSizeOfAbsExpVal[absExpVal]<mantSize: #the mantissa field of the halved cntr should be 1-bit shorter
                if mantVec[-1]=='1':
                    truncated = True
                mantVec   = mantVec[0:-1]
                mantSize -= 1
            
            if VERBOSE_DEBUG in self.verbose:
                self.cntrs[0] = self.LsbVecOfAbsExpVal[absExpVal] + mantVec
                floorVal      = self.cntr2num(self.cntrs[0])
                ceilVal       = self.incCntrBy1GetVal(forceInc=True) 
                
            if truncated and random.random()<0.5: # need to ceil the #                             
                if mantVec=='1'*mantSize: # The mantissa vector is "11...1" --> should keep the current hyperExp and exp fields, and reset the mantissa? 
                    cntr = hyperVec + expVec + '0'*(self.cntrSize - self.hyperSize - expSize)
                else:
                    mantVal = int (mantVec, base=2)
                    cntr = self.LsbVecOfAbsExpVal[absExpVal] + np.binary_repr(mantVal+1, mantSize) #[0:-1]
            else: # No need to ceil the #
                cntr = self.LsbVecOfAbsExpVal[absExpVal] + mantVec
            self.cntrs[cntrIdx] = cntr   
            if VERBOSE_DEBUG in self.verbose:
                val = self.cntr2num (cntr)
                if val==float(orgVal)/2:
                    None
                else:
                    if not (val in [floorVal, ceilVal]):
                        error ('orgVal/2={:.0f}, val={:.0f}, floorVal={:.0f}, ceilVal={:.0f}' 
                               .format (float(orgVal)/2, val, floorVal, ceilVal))
                
                
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
        for expSize in range(self.expMaxSize, 0, -1):
            hyperVec = np.binary_repr (expSize, self.hyperSize) 
            mantSize = self.cntrSize - self.hyperSize - expSize
            for i in range (2**expSize-1, 0, -1): 
                expVec = np.binary_repr(num=i, width=expSize)
                expVal = self.expVec2expVal (expVec=expVec, expSize=expSize)
                self.cntrppOfAbsExpVal[abs(expVal)] = hyperVec + np.binary_repr(num=i-1, width=expSize) + '0'*mantSize
            expVal = self.expVec2expVal (expVec='0'*expSize, expSize=expSize)
            self.cntrppOfAbsExpVal[abs(expVal)] = np.binary_repr (expSize-1, self.hyperSize) + ('1'*(expSize-1) if expSize>1 else '') + '0'*(mantSize+1)
        
    def printAllCntrs (
            self, 
            outputFile   = None,
            printAlsoVec = False, # when True, print also the counters' vectors.
        ) -> None:
        """
        Format-print all the counters as a single the array, to the given file.
        Format print the values corresponding to all the counters in self.cntrs.
        Used for debugging/logging.
        """        
        if outputFile==None:
            print (f'Printing all cntrs.')
            if printAlsoVec:
                for idx in range(self.numCntrs):
                    cntrDict = self.queryCntr (idx, getVal=False)
                    print ('cntrVec={}, cntrVal={} ' .format (cntrDict['cntrVec'], cntrDict['val']))
            else:
                for idx in range(self.numCntrs):
                    print ('{:.0f} ' .format(self.queryCntr(cntrIdx=idx, getVal=True)))
        else:
            cntrVals = np.empty (self.numCntrs)
            for idx in range(self.numCntrs):
                cntrVals[idx] = self.queryCntr(cntrIdx=idx, getVal=True)
            for cntrVal in cntrVals:
                printf (outputFile, '{:.0f} ' .format(cntrVal))
            printf (outputFile, '\n')

    def incCntr (self, cntrIdx=0, factor=int(1), mult=False, verbose=[]):
        """
        Increment the counter to the closest higher value.
        If the cntr reached its max val, or the randomization decides not to inc, merely return the cur cntr.
        Return the counter's value after the increment        
        """
        if factor==1 and mult==False:
            return self.incCntrBy1GetVal (cntrIdx=cntrIdx)
        settings.error ('In F2P_li.incCntr(). Sorry, incCntr is currently supported only for factor=1 and mult=False')
    

    def incCntrBy1GetVal (
        self, 
        cntrIdx  = 0, # idx of the concrete counter to increment in the array
        forceInc = False, # If forceInc==True, increment the counter. Else, inc the counter w.p. corresponding to the next counted value.
    ): 
        """
        Increment the counter to the closest higher value.
        If the cntr reached its max val, or the randomization decides not to inc, merely return the cur cntr.
        Return the counter's value after the increment        
        """

        cntr       = self.cntrs[cntrIdx]
        if cntr==self.cntrMaxVec: # Asked to increment a saturated counter
            return self.cntrMaxVal
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
  
