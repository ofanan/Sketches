# A single bucket of Mixed Exponent Counters.
# These counters are similar to F2P counter: 
# DIfferent ranges in the counter count using distinct, increasing, resolution.
# At the beginning, the resolution (difference between every 2 sequencing numbers) is 1.
# Later (once one of the counters is saturated), the resolution is 1 only for the smallest numbers, but 2 for larger numbers.
# Later, the resolutions are 1, 2, and 4, and so on.
# Updating the ranges of the exponents, and tuning the new counters values accordingly, is done in the function upScale.

import math, random, pickle
from printf import printf
import settings
import numpy as np

def precomputeExpRangesAndOffsets (cntrSize, numStages):
    """
    Pre-compute the expRanges (ranges corresponding to required exponent value) and the offset of each stage.
    stage holds the number of upScale happened. The initial stage is 0, and each time a counter is saturated, the stage is incremented.
    expRanges[s][i] will hold the counter's value after which the resolution is doubled (the exponent is incremented by 1) for stage s.
    For instance, if expRanges[3][1]=7 and expRanges[3][2]=15, then in stage 3, the difference between the counters' values 7 (00..0111) and 15 (00...01111) have resolution of 2**1=1 between each 2 sequencing values.   
    offsets[s][e] will hold the offset to be added to the counter's val when the stage is s and the exponent's value is e.
    """
    cntrMaxVal = int ((1 << cntrSize) - 1)
    expRanges, offsets = [[]]*numStages, [[]]*numStages #[[None]]*numStages, [[None]]*numStages
    if cntrSize<=8:
        pivots = np.zeros(numStages, dtype='uint8') 
    elif cntrSize<=16:
        pivots = np.zeros(numStages, dtype='uint16') 
    else:
        pivots = np.zeros(numStages, dtype='uint32')             
    expRanges[0] = [int(0), int(cntrMaxVal+1)]
    offsets  [0] = expRanges[0].copy ()

    for stage in range (1, numStages):
        expRanges[stage]    = expRanges[stage-1].copy ()
        # pivots [stage] will hold the new expRange added at this stage
        # Only counters larger than the pivot are affected during an upScale. 
        # Smaller counters are not affected, because smaller exponent ranges are unchanged. 
        pivots [stage]       = int((2*(stage - 2**(math.floor(math.log2(stage))))+1)/(2**(math.ceil(math.log2(stage+1))))*(cntrMaxVal+1))
        expRanges[stage].append   (pivots[stage]) # add the new expRange for this stage
        expRanges[stage].sort     () # sort the expRanges in an incresing fachion
        offsets[stage] = [int(0)]*len(expRanges[stage])
        for i in range(1, len(expRanges[stage])):
            offsets[stage][i] = offsets[stage][i-1] + (expRanges[stage][i] - expRanges[stage][i-1])*(2**(i-1))
    return expRanges, offsets, pivots

class CntrMaster (object):
    """
    Generate, check and parse counters
    """
    # Generates a strings that details the counter's settings (param vals).    
    genSettingsStr = lambda self : 'MEC_n{}_s{}' .format (self.cntrSize, self.stage)
    
    # returns the value of a number given its offset, exp and mant
    # valOf = lambda self, cntr : offset + mantVal*2**expVal
    
    def calcProbOfInc1 (self):
        """
        Calculate the array self.probOfInc1, which is defined as follows.
        self.probOfInc1[i] = the prob' of incrementing the counter by 1, when the value of the cntr is i.
        This is calculated as: self.probOfInc1[i] = 1/(value_of_the_cntr_if_incremented_by_1 - curCntrVal) 
        """
        return
    
    def __init__ (self, 
                  cntrSize  = 8, # bits per counter 
                  numCntrs  = 1, # number of counters in the bucket 
                  verbose   =[], # verbose (output) definitions, defined in settings.py.
                  ):
        
        """
        Initialize an array of MEC counters. The cntrs are initialized to 0.
        """
        if (cntrSize<3):
            settings.error (f'MecBucket was called with cntrSize={cntrSize}. However, cntrSize should be at least 3.')
        self.cntrSize   = int(cntrSize)
        self.numCntrs   = int(numCntrs)
        self.verbose    = verbose
        self.stage      = 0
        self.rstAllCntrs    ()
            
    def rstAllCntrs (self):
        """
        """
        if self.cntrSize<=8:
            self.cntrs      = np.zeros(self.numCntrs, dtype='uint8') 
        elif self.cntrSize<=16:
            self.cntrs      = np.zeros(self.numCntrs, dtype='uint16') 
        else:
            self.cntrs      = np.zeros(self.numCntrs, dtype='uint32')             
        return            
        
    def rstCntr (self, cntrIdx=0):
        """
        """
        self.cntrs[cntrIdx] = 0
        

    def cntr2val (self, 
                  cntr,         # integer representation of the counter's vec
                  stage = None  # stage to be used when calculating the value
                  ):
        """
        Given a MEC , return the value it represents and the first expRange >= this cntr. 
        Outputs: 
        - The value represented by this MEC, at this stage. 
        - The minimal expRangesIdx satisfying CntrMaster.expRanges[stage][expRangeIdx]>=cntr.
        The value is computed as the value of offset of the largest expRange still below this counter + the value to be added for the gap between the offset and the counter.
        """
        if stage==None:
            stage = self.stage
        for expRangeIdx in range(1, len(CntrMaster.expRanges[stage])):
            if CntrMaster.expRanges[stage][expRangeIdx] >= cntr: # Is this the first expRanges larger than this counter?
                # Yep. So take the offset of 1 below this expRange, which is the LARGEST expRange still below this counter. 
                return CntrMaster.offsets[stage][expRangeIdx-1] + (cntr - CntrMaster.expRanges[stage][expRangeIdx-1])*(2**(expRangeIdx-1)), expRangeIdx
        settings.error (f'in cntr2val. cntr={cntr}, self.cntrMaxVal={CntrMaster.expRanges[stage][-1]-1}, max expRanges={CntrMaster.expRanges[stage][-1]}')
        
        
    def queryCntrGetVal (self, cntrIdx=0):
        """
        Query a cntr.
        Input: 
        cntrIdx - the counter's index. 
        Output:
        cntrDic: a dictionary, where: 
            - cntrDict['cntrVec'] is the counter's binary representation; cntrDict['val'] is its value.        
        """
        return self.cntr2val(self.cntrs[cntrIdx])    

    def incCntrBy1GetVal (self, 
                    cntrIdx  = 0) ->int: # idx of the concrete counter to increment in the array
        """
        Perform probabilistic increment of 1 to the counter to the closest higher value including upscale, if needed.
        Probabilistic increment is done with prob' 1/(newValue-curValue).
        Return:
        - the value after increment.
        """
        if self.cntrs[cntrIdx]<CntrMaster.expRanges[self.stage][0]: # is the counter within a range of exponent==0?
            self.cntrs[cntrIdx] += 1 # yep --> increment by 1 and return the updated value
            return self.cntrs[cntrIdx]
        if self.cntrs[cntrIdx]==CntrMaster.expRanges[self.stage][-1]-1: # OF
            self.upScale ()
        val, expRangeIdx = self.cntr2val (self.cntrs[cntrIdx])
        if self.cntrs[cntrIdx] == CntrMaster.expRanges[self.stage][expRangeIdx]: # the cntr is exactly at the beginning (lowest value) of an expRange
            valpp = val + 2**expRangeIdx
        else:
            valpp = val + 2**(expRangeIdx-1)
            
        if random.random() < 1/(valpp-val):
            self.cntrs[cntrIdx] += 1 # yep --> increment by 1 and return the updated value
            return valpp
        return val
            
    def upScale (self):
        """
        scale-up all the counters in the bucket, by updating the exponent ranges and modifying all the cntrs accordingly.
        """
        
        if self.stage==CntrMaster.numStages-1:
            settings.error ('MecBucket: requested to upScale above the highest stage.')
        
        if settings.VERBOSE_LOG in self.verbose:
            printf (self.logFile, f'upScsale. stage={self.stage}\n')
        self.stage += 1

        for cntrIdx in range(self.numCntrs):
            
            if self.cntrs[cntrIdx]<=CntrMaster.pivots[self.stage]: # need not change any counter below the pivot
                continue
            # val, expRangeIdx will hold the value and range of cntr in the pre-upScaled array.
            val, expRangeIdx = self.cntr2val(self.cntrs[cntrIdx], self.stage-1)
            
            # Calculate the representation corresponding to val in the upScaled
            # loop on the list of offsets downwards, from expRangeIdx until reaching an offset <= val 
            for i in range(expRangeIdx+1, 0, -1):
                if CntrMaster.offsets[self.stage][i] > val: # did not reach yet an offset lower than val
                    continue
                if CntrMaster.offsets[self.stage][i]==val: # Bingo
                    self.cntrs[cntrIdx] = CntrMaster.expRanges[self.stage][i]
                    break
                
                # reached an offset<val.
                shift = (val-CntrMaster.offsets[self.stage][i])//(2**i)
                cntrVal = CntrMaster.offsets[self.stage][i] + shift*(2**i) # value of the suggested modified cntr 
                self.cntrs[cntrIdx] = CntrMaster.expRanges[self.stage][i] + shift
                if cntrVal!=val and random.random() > 0.5: # did not find exact match and need to inc 
                    self.cntrs[cntrIdx] += 1
                break 
        
    def printAllPossibleVals (self):
        """
        print all the values that can be represented at this stage.
        Used for debugging/logging.
        """
        print ([self.cntr2val(i)[0] for i in range(CntrMaster.expRanges[self.stage][-1]+1)])
        
    def incCntr (self, cntrIdx=0, mult=False, factor=1, verbose=[]):
        """
        Increase a counter by a given factor.
        Input:
        cntrIdx - index of the cntr to increment, in the array of cntrs.
        mult - if true, multiply the counter by factor. Else, increase the counter by factor.
        factor - the additive/multiplicative coefficient.
        verbose - determines which data will be written to the screen.
        Output:
        cntrDict: a dictionary representing the modified counter where: 
            - cntrDict['cntrVec'] is the counter's binary representation; cntrDict['val'] is its value.
        Operation:
        Define cntrVal as the current counter's value. 
        Then, targetValue = cntrVal*factor (if mult==True), and targetValue = cntrVal + factor (otherwise).  
        If targetValue > maximum cntr's value, return the a cntr representing the max possible value. 
        If targetValue < 0, return a cntr representing 0.
        If targetValue can be represented correctly by the counter, return the exact representation.
        Else, use probabilistic cntr's modification.
        
        If verbose==settings.VERBOSE_DETAILS, the function will print to stdout:
        - the target value (the cntr's current value + factor)
        - optionalModifiedCntr - an array with entries, representing the counters closest to the target value from below and from above.
          If the target value can be accurately represented by the counter, then optionalModifiedCntr will include 2 identical entries. 
          Each entry in optionalModifiedCntr is a cntrDict that consists of: 
          - cntrDict['cntrVec'] - the binary counter.
          - cntrDict['val']  - the counter's value.
        """
        settings.error ('F2P_bucket.incCntr() is currently unsupported.')

    def val2cntr (self, targetVal) -> list:
        """
        given a target value, find the closest counters to this targetVal from below and from above.
        Output:
        - A dictionary where 'cntrVec' is the binary counter, 'val' is its integer value.
        - If an exact match was found (the exact targetVal can be represented), the dict is the cntr representing this targetVal. 
        - If targetVal <= 0, the list has a single dict entry: the cntr representing 0 
        - If targetVal > maxVal that this cntr can represent, the dict is the cntr repesenting maxVal
        - Else, 
            The cosrresponding counter's value, after performing a probabilistic increment. 
        """
        # as the list is of offset values is sorted, need to loop only until the max relevant offset is found
        for i in range(1, len(CntrMaster.offsets[self.stage])):
            if CntrMaster.offsets[self.stage][i]<targetVal: # did not reach yet the largest relevant offset
                continue
            if CntrMaster.offsets[self.stage][i]==targetVal: # Bingo
                return [{'cntr' : CntrMaster.expRanges[self.stage][i], 'val' : targetVal}] # CntrMaster.expRanges[i] holds the counter corresponding to this offset
            
            # now we know that self.offset[i]>val. Therefore the max relevant offset is the previous one, namely, CntrMaster.offsets[self.stage][i-1]
            shift = (targetVal-CntrMaster.offsets[self.stage][i-1])//(2**(i-1))
            cntr = CntrMaster.expRanges[self.stage][i-1] + shift
            cntrVal = CntrMaster.offsets[self.stage][i-1] + shift*(2**(i-1)) 
            if cntrVal==targetVal:
                return [{'cntr' : cntr, 'val' : targetVal}]
            return [{'cntr' : cntr, 'val' : cntrVal}, {'cntr' : cntr+1, 'val' : cntrVal + 2**(i-1)}]
        

    def calcCntrMaxVal (self):
        """
        sets self.cntrMaxVal to the maximum value that may be represented by this F2P cntr. 
        """

        self.cntrZeroVec = np.binary_repr   (2**self.cntrSize - 2**(self.cntrSize-self.hyperExpSize-self.expMaxSize), self.cntrSize) # the cntr that reaches the lowest value (zero)
        # self.cntrMaxVec  = np.binary_repr   (2**(self.cntrSize-self.hyperExpSize)-1, self.cntrSize) # the cntr that reaches the highest value
        
        
    def printAllCntrVals (self, outputFile=None, printAlsoVec=False) -> None:
        """
        Format print the values corresponding to all the counters in self.cntrs.
        Used for debugging/logging.
        """        
        if outputFile==None:
            print (f'Printing all cntrs.')
            if printAlsoVec:
                for cntr in self.cntrs:
                    print (f'cntrVec={cntr}, cntrVal={self.cntr2val(cntr)[0]} ')
            else:
                for cntr in self.cntrs:
                    print (f'{self.cntr2val(cntr)[0]} ')
        else:
            for cntr in self.cntrs:
                printf (outputFile, f'{self.cntr2val(cntr)[0]} ')
    

def printAllCntrMaxVals (hyperExpSizeRange=None, cntrSizeRange=[], verbose=[settings.VERBOSE_LOG]):
    """
    print the maximum value a cntr reach for several "configurations" -- namely, all combinations of cntrSize and hyperExpSize. 
    """
    return
