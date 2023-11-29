# Mixed Exponent COunters
import math, random, pickle
from printf import printf
import settings
import numpy as np

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
    
    def calcOffsets (self):
        """
        Pre-calculate all the offsets to be added to a counter, according to its exponent value.
        self.offsetOfExpVal[e] will hold the offset to be added to the counter's val when the exponent's value is e.
        """
        return

   
    def __init__ (self, 
                  cntrSize  = 8, # bits per counter 
                  stageSize = 4, # bits of the "stage" field in each bucket 
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
        self.cntrMaxVal = (1 << self.cntrSize) - 1
        self.verbose    = verbose
        self.stage      = 0
        self.stageMax   = (1 << stageSize) - 1
        self.expRanges  = [0, self.cntrMaxVal+1]
        self.rstAllCntrs ()
        for _ in range (15): #$$$
            self.scaleUp ()

        
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
        

    def cntr2val (self, cntr):
        """
        Convert a MEC, given as an integer, to the value it represents.
        """
        # if cntr<self.expRanges[1]:
        #     return cntr
        val = 0
        for expRangeIdx in range(1, len(self.expRanges)):
            if self.expRanges[expRangeIdx] >= cntr:
                val += (cntr - self.expRanges[expRangeIdx-1])*(2**(expRangeIdx-1))
                break
            val += (self.expRanges[expRangeIdx]-self.expRanges[expRangeIdx-1])*(2**(expRangeIdx-1))
        return val

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
                    cntrIdx  = 0): # idx of the concrete counter to increment in the array
        """
        Increment the counter to the closest higher value.
        If the counter is already the maximal value, do nothing.
        Else, increment the counter with prob' 1/(newValue-curValue).
        Return:
        - the value after increment.
        """
        if self.cntrs[cntrIdx]<self.expRanges[0]: # is the counter within a range of exponent==0?
            self.cntrs[cntrIdx] += 1 # yep --> increment by 1 and return the updated value
            return self.cntrs[cntrIdx]
        if self.cntrs[cntrIdx] < self.cntrMaxVal: # No OF
            cntrVal = self.cntr2val(self.cntrs[cntrIdx])
            cntrValpp = self.cntr2val(self.cntrs[cntrIdx] + 1)
            if random.random() < 1/(cntrValpp-cntrVal): # Prob' Increment
                self.cntrs[cntrIdx] += 1
                return cntrValpp
            return cntrVal # don't increment --> return the current value, w/o increment
        self.scaleUp ()
        self.cntrs[cntrIdx] += 1
        return self.cntr2val(self.cntrs[cntrIdx])
            
    def scaleUp (self):
        """
        scale-up all the counters in the bucket, by updating the exponent ranges and halving counters.
        """
        if self.stage==self.stageMax:
            settings.error ('MecBucket: cannot upScale above the maximum stage.')
        self.stage += 1
        # j = self.stage - 2**(math.floor(math.log2(self.stage)))
        # nom = 2*j+1
        # denom = 2**(math.ceil(math.log2(self.stage+1)))
        # frac = nom/denom
        # frac = (2*(self.stage - 2**(math.floor(math.log2(self.stage))))+1)/(2**(math.ceil(math.log2(self.stage+1))))
        pivot = int((2*(self.stage - 2**(math.floor(math.log2(self.stage))))+1)/(2**(math.ceil(math.log2(self.stage+1))))*(self.cntrMaxVal+1))
        self.expRanges.append (pivot)
        self.expRanges.sort()
        # print (f'stage={self.stage}, j={j}, frac={nom}/{denom}, frac={frac}, expRanges={self.expRanges}')
        print (f'stage={self.stage}, expRanges={self.expRanges}')
        
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
            
            
    def num2cntr (self, targetVal) -> dict:
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
        return
        

    def calcCntrMaxVal (self):
        """
        sets self.cntrMaxVal to the maximum value that may be represented by this F2P cntr. 
        """

        self.cntrZeroVec = np.binary_repr   (2**self.cntrSize - 2**(self.cntrSize-self.hyperExpSize-self.expMaxSize), self.cntrSize) # the cntr that reaches the lowest value (zero)
        self.cntrMaxVec  = np.binary_repr   (2**(self.cntrSize-self.hyperExpSize)-1, self.cntrSize) # the cntr that reaches the highest value
        self.cntrMaxVal  = self.cntr2val (self.cntrMaxVec) 
        
    def printCntrs (self, outputFile=None, printAlsoVec=False) -> None:
        """
        Format-print all the counters as a single the array, to the given file.
        """
        return
        # if outputFile==None:
        #     print (f'Printing all cntrs.')
        #     if printAlsoVec:
        #         for cntr in self.cntrs:
        #             print (f'cntrVec={cntr}, cntrVal={self.cntr2val(cntr)} ')
        #     else:
        #         for cntr in self.cntrs:
        #             print (f'{self.cntr2val(cntr)} ')
        # else:
        #     for cntr in self.cntrs:
        #         printf (outputFile, f'{self.cntr2val(cntr)} ')
    

def printAllCntrMaxVals (hyperExpSizeRange=None, cntrSizeRange=[], verbose=[settings.VERBOSE_LOG]):
    """
    print the maximum value a cntr reach for several "configurations" -- namely, all combinations of cntrSize and hyperExpSize. 
    """
    return
