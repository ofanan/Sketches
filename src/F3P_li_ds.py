# This file implements F3P LI_ds, namely, F3P flavor that represents only integers, and focuses on improved accuracy on elephants; and supports down-sampling.
# This class mainly inherits from the class F3P_li. 
# For further details, see "main.tex" in Cntr's Overleaf project.
import math, random, pickle, numpy as np

from printf import printf
import settings, F3P_li
from settings import * 
np.set_printoptions(precision=4)

class CntrMaster (F3P_li.CntrMaster):
    """
    Generate, check and perform arithmetic operations on F3P counters in LI flavors with down-sampling.
    The counters are generated as an array with the same format (counterSize and hyperExpSize).
    Then, it's possible to perform arithmetic ops on the counters in chosen indices of the array. 
    """
    
    cntrMaxValWDwnSmpl = lambda self : self.cntrMaxVal/self.globalIncProb
    
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
        if VERBOSE_DEBUG in verbose:
            self.maxValAtLastUpSclae = 0
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
            absExpVal = abs(self.expVec2expVal(expVec, expSize))
            mantSize  = len(mantVec) 
            orgMantSize = mantSize 

            # Need to code the special case of (sub) normal values.
            probOfCeil = 0 # By default, we didn't truncate the # when dividing by 2 --> no need to round. 
            
            if absExpVal==self.Vmax-1: # The edge case of sub-normal values: need to only divide the mantissa; no need (and cannot) further decrease the exponent
                if mantVec[-1]=='1':
                    probOfCeil = 0.5 # LSB was '1' --> will later round-up the halved value w.p. 0.5
                mantVec = '0' + mantVec[0:-1] # mantVec >> 1 # divide the mantissa by 2 (by right-shift) 
            elif absExpVal==self.Vmax-2: # The edge case of 1-above sub-normal values: need to right-shift the mantissa, inserting '1' into the new leftmost mantissa bit. 
                if mantVec[-1]=='1':
                    probOfCeil = 0.5 # LSB was '1' --> will later round-up the halved value w.p. 0.5
                mantVec = '1' + mantVec[0:-1]                
            elif self.mantSizeOfAbsExpVal[absExpVal]<mantSize: #the mantissa field of the halved cntr is shorter than the current mantissa field 
                mantSizeDiff = mantSize - self.mantSizeOfAbsExpVal[absExpVal] # The # of bits to deduce from the older mantissa field to get the new mantissa field.
                if mantSizeDiff>0: 
                    probOfCeil   = float (int (mantVec[-mantSizeDiff:], base=2)) / 2**mantSizeDiff # the mantSizeDiff LSBs of the mantissa reflect the prob' to round-up (ceil) the value after the shift-right we're about to perform 
                    mantVec      = mantVec[0:-mantSizeDiff] # shift-right the mantissa mantSizeDiff places  
                    mantSize    -= mantSizeDiff # size of the new mantissa
                
            floorCntr = self.LsbVecOfAbsExpVal[absExpVal] + mantVec # Re-build the (floor) half counter from the (pre-computed) hyper-exp [optional: delimiter] and exp field, followed by the new mantissa.
            if probOfCeil>0: # have to round-up (ceil) with some non-zero probability --> calculate the ceil value.
                if mantVec=='1'*mantSize: # The mantissa vector is "11...1" --> keep the current hyperExp and exp fields, and reset the mantissa 
                    ceilCntr = cntr[0:-orgMantSize] + '0'*orgMantSize
                else: # inc. the mantissa
                    ceilCntr = self.LsbVecOfAbsExpVal[absExpVal] + np.binary_repr(int (mantVec, base=2)+1, mantSize) #[0:-1]
            else:
                ceilCntr = floorCntr
            if VERBOSE_LOG_DWN_SMPL_D in self.verbose:
                printf (self.logFile, f'cntr={cntr}, floorCntr={floorCntr}, ceil={ceilCntr}, probOfCeil={probOfCeil}, expVec={expVec}, absExpVal={absExpVal}, ')
                orgVal   = self.cntr2num(cntr)
                floorVal = self.cntr2num(floorCntr)
                ceilVal  = self.cntr2num(ceilCntr)
                printf (self.logFile,  'orgVal={:.0f}, floorVal={:.0f}, ceilVal={:.0f}\n'
                        .format (orgVal, floorVal, ceilVal))
                if probOfCeil>0 and probOfCeil!=float(orgVal/2-floorVal)/float(ceilVal-floorVal):
                    error ('In F3P_li_ds.upScale(). suspected wrong probability calculation. Please check the log file under ../res/log_files.')
            if probOfCeil>0 and random.random()<probOfCeil>0: # need to ceil the #                             
                self.cntrs[cntrIdx] = ceilCntr
            else: 
                self.cntrs[cntrIdx] = floorCntr
            if len(self.cntrs[cntrIdx])>self.cntrSize:
                error (f'In F3P_li_ds. curCntr={self.cntrs[cntrIdx]}. upScaledCntr={cntr}')
                
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
            if VERBOSE_DEBUG in self.verbose:
                if self.maxValAtLastUpSclae>self.cntrMaxValWDwnSmpl():
                    self.printAllCntrs ()
                    error (f'at last upscaling maxVal={self.maxValAtLastUpSclae}. Now maxVal={self.cntrMaxValWDwnSmpl()}. globalIncProb={self.globalIncProb}')
                self.maxValAtLastUpSclae = self.cntrMaxValWDwnSmpl()
                print (f'when upscaling, maxVal={self.maxValAtLastUpSclae}')                
            if self.cntr2num(self.cntrs[cntrIdx])!=self.cntrMaxValWDwnSmpl():
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
        
        probOfInc1 = self.probOfInc1[abs(expVal)]

        if (random.random() > probOfInc1):  # check first the case where we don't have to inc the counter 
            return int(cntrCurVal)    

        # now we know that we have to inc. the cntr
        cntrppVal  = cntrCurVal + (1/probOfInc1)
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
