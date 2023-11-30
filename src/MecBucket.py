# Mixed Exponent COunters
import math, random, pickle
from printf import printf
import settings
import numpy as np

def precomputeExpRangesAndOffsets (cntrSize, numStages):

    cntrMaxVal = int ((1 << cntrSize) - 1)

    expRanges, offsets = [[]]*numStages, [[]]*numStages #[[None]]*numStages, [[None]]*numStages
    # print (expRanges)
    expRanges[0] = [int(0), int(cntrMaxVal+1)]
    offsets  [0] = expRanges[0].copy () 
    # settings.error (expRanges)

    # expRanges.append ([int(0), int(cntrMaxVal+1)])
    # offsets.  append ([int(0), int(cntrMaxVal+1)])
    # expRanges.append ([int(0), int(cntrMaxVal+1)])
    # , offsets[0] = [int(0), int(cntrMaxVal+1)] 
    # settings.error (expRanges)
    # if cntrSize<=8:
    #     expRanges[0]    = np.zeros(2, dtype='uint8')
    #     offsets[0]      = np.zeros(2, dtype='uint64')
    # elif cntrSize<=16:
    #     expRanges[0]    = np.zeros(2, dtype='uint16')
    #     offsets[0]      = np.zeros(2, dtype='uint64')
    # else:
    #     expRanges[0]    = np.zeros(2, dtype='uint32')
    #     offsets[0]      = np.zeros(2, dtype='uint64')

    for stage in range (1, numStages):
        pivot = int((2*(stage - 2**(math.floor(math.log2(stage))))+1)/(2**(math.ceil(math.log2(stage+1))))*(cntrMaxVal+1))
        # prevOffsets = self.offsets.copy()
        # settings.error (expRanges[stage-1])
        # settings.error  (f'stage={stage}, expRanges[stage-1]={expRanges[stage-1]}, expRanges={expRanges}')
        expRanges[stage] = expRanges[stage-1].copy ()
        expRanges[stage].append   (pivot)
        expRanges[stage].sort     ()
        offsets[stage] = [int(0)]*len(expRanges[stage])
        for i in range(1, len(expRanges[stage])):
            offsets[stage][i] = offsets[stage][i-1] + (expRanges[stage][i] - expRanges[stage][i-1])*(2**(i-1))

    print (f'expRanges={expRanges}, offsets={offsets}')
    exit ()
    return expRanges
        # self.updateOffsets      ()
        
        
    # expRanges[0] = [int(0), cntrMaxVal+1]
    #
    # self.updateOffsets  ()
    # self.rstAllCntrs    ()
    #
    # offsets = [int(0)]*len(self.expRanges)
    # for i in range(1, len(self.expRanges)):
    #     self.offsets[i] = self.offsets[i-1] + (self.expRanges[i] - self.expRanges[i-1])*(2**(i-1))
    # print (f'expRanges={self.expRanges}, offsets={self.offsets}') #$$$


class CntrMaster (object):
    """
    Generate, check and parse counters
    """
    expRanges = None

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
                  stageSize = 4, # bits of the "stage" field in each bucket 
                  numCntrs  = 1, # number of counters in the bucket 
                  verbose   =[], # verbose (output) definitions, defined in settings.py.
                  ):
        
        """
        Initialize an array of MEC counters. The cntrs are initialized to 0.
        """
        settings.error (CntrMaster.expRanges) #$$
        exit () #$$
        if (cntrSize<3):
            settings.error (f'MecBucket was called with cntrSize={cntrSize}. However, cntrSize should be at least 3.')
        self.cntrSize   = int(cntrSize)
        self.numCntrs   = int(numCntrs)
        self.cntrMaxVal = int ((1 << self.cntrSize) - 1)
        self.verbose    = verbose
        self.stage      = 0
        self.stageMax   = int ((1 << stageSize) - 1)
        self.expRanges  = [int(0), self.cntrMaxVal+1]
        self.updateOffsets  ()
        self.rstAllCntrs    ()
        for _ in range (5): #$$
            self.upScale ()
            
        
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
                  cntr, # integer representation of the counter's vec
                  expRanges = None # expRanges to use during the calculation
                  # offsets = None # offsets to use during the calculation
                  ):
        """
        Convert a MEC to the value it represents.
        """
        # if offsets==None:
        #     offsets = self.offsets
        if expRanges==None:
            expRanges = self.expRanges
        for expRangeIdx in range(1, len(expRanges)):
            if expRanges[expRangeIdx] >= cntr:
                val += (cntr - expRanges[expRangeIdx-1])*(2**(expRangeIdx-1))
                break
            val += (expRanges[expRangeIdx]-expRanges[expRangeIdx-1])*(2**(expRangeIdx-1))
        return val

    def updateOffsets (self):
        """
        Calculate the offset corresponding to each expRange
        """
        """
        Pre-calculate all the offsets to be added to a counter, according to its exponent value.
        self.offsetOfExpVal[e] will hold the offset to be added to the counter's val when the exponent's value is e.
        """
        self.offsets = [int(0)]*len(self.expRanges)
        for i in range(1, len(self.expRanges)):
            self.offsets[i] = self.offsets[i-1] + (self.expRanges[i] - self.expRanges[i-1])*(2**(i-1))
        print (f'expRanges={self.expRanges}, offsets={self.offsets}') #$$$

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
        upScaled = False #$$$
        if self.cntrs[cntrIdx]==self.cntrMaxVal: # OF
            vecb4upScale = self.cntrs[cntrIdx]
            valb4upScale = self.cntr2val(self.cntrs[cntrIdx])
            self.upScale ()
            upScaled = True #$$
        cntrVal = self.cntr2val(self.cntrs[cntrIdx])
        cntrValpp = self.cntr2val(self.cntrs[cntrIdx] + 1)
        if upScaled: #$$$
            print (f'b4upScale: vec={vecb4upScale} val={valb4upScale}. After: vec={self.cntrs[cntrIdx]}, cntrVal={cntrVal}, cntrValpp={cntrValpp}')
        if random.random() < 1/(cntrValpp-cntrVal): # Prob' Increment
            self.cntrs[cntrIdx] += 1
            return cntrValpp
        return cntrVal # don't increment --> return the current value, w/o increment
            
    def upScale (self):
        """
        scale-up all the counters in the bucket, by updating the exponent ranges and halving counters.
        """
        if self.stage==self.stageMax:
            settings.error ('MecBucket: cannot upScale above the maximum stage.')
        if settings.VERBOSE_LOG in self.verbose:
            printf (self.logFile, f'upScsale. stage={self.stage}\n')
        self.stage += 1
        pivot = int((2*(self.stage - 2**(math.floor(math.log2(self.stage))))+1)/(2**(math.ceil(math.log2(self.stage+1))))*(self.cntrMaxVal+1))
        # prevOffsets = self.offsets.copy()
        prevExpRanges = self.expRanges.copy()
        self.expRanges.append   (pivot)
        self.expRanges.sort     ()
        self.updateOffsets      ()
        return #$$$

        for cntrIdx in range(self.numCntrs):
            prevVal = self.cntr2val(self.cntrs[cntrIdx], prevExpRanges)
            cntrs = self.val2cntr (prevVal)
            if len(cntrs)==1: # could accurately represent the new counter; is that possible at all?
                settings.error ('found exact match after upScsaling') #$$$
                self.cntrs[cntrIdx] = cntrs[0]
                continue
            print (f'prevVal={prevVal}, newVal={self.cntr2val(cntrs[0])}, newValPp={self.cntr2val(cntrs[1])}') #$$$
            if random.random()<0.5:
                self.cntrs[cntrIdx] = cntrs[0]
            else:
                self.cntrs[cntrIdx] = cntrs[1]
        # self.printAllPossibleVals () #$$$$
        # print (f'stage={self.stage}, pivot={pivot}, expRanges={self.expRanges}, relevantExpRanges={relevantExpRanges}')
        
    def printAllPossibleVals (self):
        
        print (f'stage={self.stage}')
        print ([self.cntr2val(i) for i in range(self.cntrMaxVal+1)])
        
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
            
            
    # def cntr2val (self, 
    #               cntr, # integer representation of the counter's vec
    #               offsets = None # offsets to use during the calculation
    #               ):
    #     """
    #     Convert a MEC to the value it represents.
    #     """
    #     if offsets==None:
    #         offsets = self.offsets
    def val2cntr (self, targetVal) -> dict:
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
        settings.error ('val2cntr is not implemented yet')
        # as the list is of offset values is sorted, need to loop only until the max relevant offset is found
        for i in range(1, len(self.offsets)):
            if self.offsets[i]<targetVal: # did not reach yet the largest relevant offset
                continue
            if self.offsets[i]==targetVal: # Bingo
                return [{'cntr' : self.expRanges[i], 'val' : targetVal}] # expRanges[i] holds the counter corresponding to this offset
            
            # now we know that self.offset[i]>val. Therefore the max relevant offset is the previous one, namely, self.offsets[i-1]
            cntr = self.expRanges[i-1] + (targetVal-self.offsets[i-1])*(2**(i-1))
            cntrVal = self.offsets[i-1] + (cntr-self.expRanges[i-1])*(2**(i-1)) 
            if cntrVal==targetVal:
                return [{'cntr' : cntr, 'val' : targetVal}]
            return [{'cntr' : cntr, 'val' : cntrVal}, {'cntr' : cntr+1, 'val' : cantrVal + 2**(i-1)}]
        

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
