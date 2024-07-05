# This file implements F3P LI_ds, namely, F3P flavor that represents only integers, and focuses on improved accuracy on elephants; and supports down-sampling.
# This class mainly inherits from the class F3P_li. 
# For further details, see "main.tex" in Cntr's Overleaf project.
import math, random, pickle, numpy as np

from printf import printf
import settings, F3P_li
from settings import error, VERBOSE_DEBUG, VERBOSE_LOG, VERBOSE_LOG_DWN_SMPL

class CntrMaster (F3P_li.CntrMaster):
    """
    Generate, check and perform arithmetic operations on F3P counters in LI flavors with down-sampling.
    The counters are generated as an array with the same format (counterSize and hyperExpSize).
    Then, it's possible to perform arithmetic ops on the counters in chosen indices of the array. 
    """

    def __init__ (self, 
            cntrSize        : int  = 8, # of bits in the cntr 
            hyperMaxSize    : int  = 1, # of bits in the hyper-exp field 
            numCntrs        : int  = 1, # of cntrs in the cntrs' array
            verbose         : list = []    # the optional verbose values are detailed in settings.py
        ):
        
        """
        Initialize an array of cntrSize counters. The cntrs are initialized to 0.
        If the parameters are invalid (e.g., infeasible cntrSize), return None. 
        """       
        self.globalIncProb = 1.0 # Probability to consider an increment for any counter. After up-scaling, this probability decreases.
        super(CntrMaster, self).__init__ (
            cntrSize        = cntrSize, 
            hyperMaxSize    = hyperMaxSize, # of bits in the hyper-exp field 
            numCntrs        = numCntrs,
            verbose         = verbose
        )

    def setFlavorParams (self):
        """
        set variables that are unique for 'li' flavor of F3P.
        """
        super(CntrMaster, self).setFlavorParams ()
        self.LsbVecOfAbsExpVal   = ['']*(self.Vmax) # self.LsbVecOfAbsExpVal[e] will hold the LSB fields (hyperVec and expVec) of the vector when decreasing the vector's exponent whose current absolute value is e.
        for hyperSize in range(0, self.hyperMaxSize+1):
            for i in range (2**hyperSize):
                expVec = np.binary_repr(num=i, width=hyperSize) if hyperSize>0 else ''
                expVal = self.expVec2expVal (expVec=expVec, expSize=hyperSize)
                if hyperSize==self.hyperMaxSize:
                    self.LsbVecOfAbsExpVal[abs(expVal)-1] = '1'*hyperSize + expVec
                else: 
                    self.LsbVecOfAbsExpVal[abs(expVal)-1] = '1'*hyperSize + '0' + expVec
        self.LsbVecOfAbsExpVal[self.Vmax-1] = '1'*2*self.hyperMaxSize 
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
        for cntrIdx in range(self.numCntrs):
            cntr      = self.cntrs[cntrIdx]         # Extract the hyper-exponent field, and value
            
            hyperSize = settings.idxOfLeftmostZero (ar=cntr, maxIdx=self.hyperMaxSize)         
            expSize   = hyperSize
            if hyperSize < self.hyperMaxSize: # if the # of trailing max < hyperMaxSize, the cntr must have a delimiter '0'
                expVecBegin  = hyperSize+1
            else:
                expVecBegin  = self.hyperMaxSize
            expVec  = cntr[expVecBegin : expVecBegin+expSize]
            mantVec = cntr[expVecBegin+expSize:]
            expVal  = self.expVec2expVal(expVec, expSize) 
            absExpVal = abs(expVal)
            mantSize  = len(mantVec) 

            # Need to code the special case of (sub) normal values.
            if VERBOSE_DEBUG in self.verbose:
                orgVal = self.cntr2num (cntr)
            truncated = False # By default, we didn't truncate the # when dividing by 2 --> no need to round. 
            
            if absExpVal==self.Vmax-1: # The edge case of sub-normal values: need to only divide the mantissa; no need (and cannot) further decrease the exponent
                if mantVec[-1]=='1':
                    truncated = True
                mantVec = '0' + mantVec[0:-1] # mantVec >> 1 # divide the mantissa by 2 (by right-shift) 
            elif absExpVal==self.Vmax-2: # The edge case of 1-above sub-normal values: need to only right-shift the value, and insert '1' in the new leftmost mantissa bit. 
                if mantVec[-1]=='1':
                    truncated = True
                mantVec = '1' + mantVec[0:-1]                
            elif self.mantSizeOfAbsExpVal[absExpVal]<mantSize: #the mantissa field of the halved cntr should be 1-bit shorter
                if mantVec[-1]=='1':
                    truncated = True
                mantVec   = mantVec[0:-1]
                mantSize -= 1
            
            # if VERBOSE_DEBUG in self.verbose:
            #     self.cntrs[0] = self.LsbVecOfAbsExpVal[absExpVal] + mantVec
            #     floorVal      = self.cntr2num(self.cntrs[0])
            #     ceilVal       = self.incCntrBy1GetVal(forceInc=True) 
                
            if VERBOSE_LOG_DWN_SMPL in self.verbose:
                floorCntr = self.LsbVecOfAbsExpVal[absExpVal] + mantVec
                if truncated: 
                    if mantVec=='1'*mantSize: # The mantissa vector is "11...1" --> should keep the current hyperExp and exp fields, and reset the mantissa? 
                        ceilCntr = '1'*hyperSize + expVec + '0'*(self.cntrSize - hyperSize - expSize)
                    else:
                        mantVal = int (mantVec, base=2)
                        ceilCntr = self.LsbVecOfAbsExpVal[absExpVal] + np.binary_repr(mantVal+1, mantSize) #[0:-1]
                else:
                    ceilCntr = floorCntr
                printf (self.logFile, f'cntr={cntr}, floorCntr={floorCntr}, ceil={ceilCntr}, expVec={expVec}, absExpVal={absExpVal}, ')
                printf (self.logFile,  'Val={:.0f}, floorVal={:.0f}, ceilVal={:.0f}\n'
                        .format (self.cntr2num(cntr), self.cntr2num(floorCntr), self.cntr2num(ceilCntr)))
            if truncated and random.random()<0.5: # need to ceil the #                             
                if mantVec=='1'*mantSize: # The mantissa vector is "11...1" --> should keep the current hyperExp and exp fields, and reset the mantissa? 
                    cntr = '1'*hyperSize + expVec + '0'*(self.cntrSize - hyperSize - expSize)
                else:
                    mantVal = int (mantVec, base=2)
                    cntr = self.LsbVecOfAbsExpVal[absExpVal] + np.binary_repr(mantVal+1, mantSize) #[0:-1]
            else: # No need to ceil the #
                cntr = self.LsbVecOfAbsExpVal[absExpVal] + mantVec
            if len(cntr)>8:
                error (f'In F3P_li_ds. curCntr={self.cntrs[cntrIdx]}. upScaledCntr={cntr}')
            self.cntrs[cntrIdx] = cntr   
            # if VERBOSE_DEBUG in self.verbose:
            #     val = self.cntr2num (cntr)
            #     if val==float(orgVal)/2:
            #         None
            #     else:
            #         if not (val in [floorVal, ceilVal]):
            #             error ('orgVal/2={:.0f}, val={:.0f}, floorVal={:.0f}, ceilVal={:.0f}' 
            #                    .format (float(orgVal)/2, val, floorVal, ceilVal))
                
    def incCntrBy1GetVal (self, 
            cntrIdx  = 0, # idx of the concrete counter to increment in the array
        ): 
        """
        Increment the counter to the closest higher value, when down-sampling is enabled.
        If the cntr reached its max val, up-scale and perform down-sampling.  
        If the randomization decides not to inc, merely return the cur cntr.
        Return the counter's value after the increment.        
        """
        
        if self.globalIncProb<1 and random.random()>self.globalIncProb: # consider first the case where we do not need to increment the counter - only sample its value 
            return self.cntr2num (self.cntrs[cntrIdx])

        # now we know that we should consider incrementing the counter
        if self.cntrs[cntrIdx]==self.cntrMaxVec: # Asked to increment a saturated counter
            if self.cntr2num(self.cntrs[cntrIdx])!=self.cntrMaxVal/self.globalIncProb:
                error ('In F3P_li_ds.incCntrBy1GetVal(). Wrong CntrMaxVal. cntrVal={self.cntr2num(self.cntrs[cntrIdx])}self.cntr2num(self.cntrs[cntrIdx]), curCntrMaxVal={self.cntrMaxVal/self.globalIncProb}')                
            if VERBOSE_LOG_DWN_SMPL in self.verbose:
                if self.numCntrs<10:
                    printf (self.logFile, f'\nb4 upScaling:\n')
                    self.printAllCntrs (self.logFile)
                else:
                    printf (self.logFile, 'cntrVal={:.0f}. upScaling.\n' .format (self.cntr2num(self.cntrs[cntrIdx])))
            self.upScale ()
            self.globalIncProb /= 2
            if self.numCntrs<10:
                printf (self.logFile, f'\nafter upScaling:\n')
                self.printAllCntrs (self.logFile)

        if self.cntrs[cntrIdx]==self.cntrMaxVec: # Asked to increment a saturated counter
            error (f'cntr={self.cntrs[cntrIdx]} after upScaling')
            
        # Consider incrementing the counter, as usual
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
        
        # now we know that we have to inc. the cntr
        cntrppVal  = cntrCurVal + (1/self.probOfInc1[abs(expVal)])
        if mantVec == '1'*mantSize: # the mantissa overflowed
            self.cntrs[cntrIdx] = self.cntrppOfAbsExpVal[abs(expVal)]
        else:
            if hyperSize<self.hyperMaxSize:
                self.cntrs[cntrIdx] = '1'*hyperSize + '0' + expVec + np.binary_repr(num=mantIntVal+1, width=mantSize) 
            else:
                self.cntrs[cntrIdx] = '1'*hyperSize       + expVec + np.binary_repr(num=mantIntVal+1, width=mantSize) 
        return int(cntrppVal) 

# myCntr = CntrMaster (
#     cntrSize     = 4, 
#     hyperMaxSize = 1,
#     verbose=[VERBOSE_LOG_DWN_SMPL, VERBOSE_LOG]
# )
# logFile = open ('../res/log_files/F3P_li_ds.log', 'w')
# myCntr.setLogFile (logFile)
# for _ in range (100):
#     myCntr.incCntrBy1GetVal ()
