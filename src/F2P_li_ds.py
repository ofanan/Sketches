# This file implements F2P LI DS, namely, F2P flavor that represents only integers, and focuses on improved accuracy on elephants; and allows down-sampling.
# The class mainly inherits from the class F2P_li. 
# For further details, see "main.tex" in Cntr's Overleaf project.
import math, random, pickle, numpy as np

from printf import printf
import settings, F2P_li
from settings import VERBOSE_DEBUG, VERBOSE_LOG, VERBOSE_LOG_DWN_SMPL
from settings import warning, error

class CntrMaster (F2P_li.CntrMaster):
    """
    Generate, check and perform arithmetic operations of the counters. 
    The counters are generated as an array with the same format (counterSize and hyperExpSize).
    Then, it's possible to perform arithmetic ops on the counters in chosen indices of the array. 
    """

    def __init__ (self, 
            cntrSize  : int = 8, # of bits in the cntr 
            hyperSize : int = 1, # of bits in the hyper-exp field 
            numCntrs  : int = 1, # of cntrs in the cntrs' array
            verbose   = []    # the optional verbose values are detailed in settings.py
        ):
        
        """
        Initialize an array of cntrSize counters. The cntrs are initialized to 0.
        If the parameters are invalid (e.g., infeasible cntrSize), return None. 
        """       
        self.globalIncProb = 1.0 # Probability to consider an increment for any counter. After up-scaling, this probability decreases.
        super(CntrMaster, self).__init__ (
            cntrSize    = cntrSize, 
            numCntrs    = numCntrs, 
            verbose     = verbose)

    def setFlavorParams (self):
        """
        set variables that are unique for 'li' flavor of F2P.
        """
        super(CntrMaster, self).setFlavorParams ()
        self.LsbVecOfAbsExpVal   = ['']*(self.Vmax) # self.LsbVecOfAbsExpVal[e] will hold the LSB fields (hyperVec and expVec) of the vector when decreasing the vector's exponent whose current absolute value is e.
        for expSize in range(self.expMaxSize, 0, -1):
            hyperVec = np.binary_repr (expSize, self.hyperSize) 
            mantSize = self.cntrSize - self.hyperSize - expSize
            for i in range (2**expSize-1, 0, -1): 
                expVec = np.binary_repr(num=i, width=expSize)
                expVal = self.expVec2expVal (expVec=expVec, expSize=expSize)
                self.LsbVecOfAbsExpVal[abs(expVal)-1] = hyperVec + expVec 
            expVal = self.expVec2expVal (expVec='0'*expSize, expSize=expSize)
            self.LsbVecOfAbsExpVal[abs(expVal)-1] = hyperVec + '0'*expSize
        self.LsbVecOfAbsExpVal[self.Vmax-1] = '1'*(self.hyperSize + 2**self.hyperSize - 1) 
        self.mantSizeOfAbsExpVal = [self.cntrSize - len(item) for item in self.LsbVecOfAbsExpVal] # self.mantSizeOfAbsExpVal[e] is the size of the mantissa field of the vector when decreasing the vector's exponent whose current absolute value is e.
        
    def cntr2num (self, cntr):
        """
        Given a counter, as a binary vector (e.g., "11110"), return the number it represents.
        """
        return super(CntrMaster, self).cntr2num (cntr=cntr)/self.globalIncProb 

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
                
    def incCntrBy1GetVal (
        self, 
        cntrIdx  = 0, # idx of the concrete counter to increment in the array
    ): 
        """
        Increment the counter to the closest higher value, when down-sampling is enabled.
        If the cntr reached its max val, up-scale and perform down-sampling.  
        If the randomization decides not to inc, merely return the cur cntr.
        Return the counter's value after the increment.        
        """

        if self.globalIncProb<1 and random.random()<self.globalIncProb: # consider first the case where we do not need to increment the counter - only sample its value 
            return self.cntr2num (self.cntrs[cntrIdx])

        # now we know that we should consider incrementing the counter
        if self.cntrs[cntrIdx]==self.cntrMaxVec: # Asked to increment a saturated counter
            if VERBOSE_LOG_DWN_SMPL in self.verbose:
                printf (self.logFile, f'\nb4 upScaling:\n')
                self.printAllCntrs (self.logFile)
            self.upScale ()
            self.globalIncProb /= 2
            if VERBOSE_LOG_DWN_SMPL in self.verbose:
                printf (self.logFile, f'\nafter upScaling:\n')
                self.printAllCntrs (self.logFile)

        if self.cntrs[cntrIdx]==self.cntrMaxVec: # Asked to increment a saturated counter
            error (f'cntr={self.cntrs[cntrIdx]} after upScaling')
            
        # Consider incrementing the counter, as usual
        cntr = self.cntrs[cntrIdx]
        hyperVec    = cntr [0:self.hyperSize]
        expSize     = int(hyperVec, base=2)
        expVec      = cntr[self.hyperSize:self.hyperSize+expSize]
        expVal      = int (self.expVec2expVal(expVec, expSize))
        mantSize    = self.cntrSize - self.hyperSize - expSize 
        mantVec     = cntr[-mantSize:]
        mantIntVal  = int (mantVec, base=2)
        mantVal     = float (mantIntVal) / 2**(self.cntrSize - self.hyperSize - expSize)  

        if expVec == self.expMinVec:
            cntrCurVal = mantVal * (2**self.powerMin)
        else:
            cntrCurVal = (1 + mantVal) * (2**(expVal+self.bias))
        
        cntrCurVal /= self.globalIncProb
        if (random.random() > self.probOfInc1[abs(expVal)]): 
            return int(cntrCurVal)    

        # now we know that we have to inc. the cntr
        if mantVec == '1'*mantSize: # the mantissa overflowed
            self.cntrs[cntrIdx] = self.cntrppOfAbsExpVal[abs(expVal)]
            if len(self.cntrs[cntrIdx])!=self.cntrSize:
                error (f'1. cntrSize={len(self.cntrs[cntrIdx])}')
        else:
            self.cntrs[cntrIdx] = hyperVec + expVec + np.binary_repr(num=mantIntVal+1, width=mantSize) 
            if len(self.cntrs[cntrIdx])!=self.cntrSize:
                error (f'2. cntrSize={len(self.cntrs[cntrIdx])}')
        return self.cntr2num(self.cntrs[cntrIdx])
#     cntrSize    = 10, 
#     hyperSize   = 2,
#     verbose     = [VERBOSE_DEBUG]
# ) 
# logFile = open (f'../res/log_files/{myF2P_li_cntr.genSettingsStr()}.log', 'w')
# myF2P_li_cntr.setLogFile (logFile)
# myF2P_li_cntr.upScale()