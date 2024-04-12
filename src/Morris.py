#import itertools
# import time, random, sys, os
# from   pathlib import Path
# from builtins import True False
# import pickle
import math, time, random
from printf import printf
import settings
import numpy as np

# The 'a' parameter determines Morris counter's accuracy.
# Given a counter size and maximum value to count, the function estimateAGivenCntrSize finds the optimal a. using binary search.
# To prevent overflows, the search range should be limited. 
# This is done using the list of dicts below. 
aSearchRanges = [
                 {'cntrSize' : 5,    'aLo' : 1,      'aHi' : 1000},
                 {'cntrSize' : 6,    'aLo' : 1,      'aHi' : 1000},
                 {'cntrSize' : 7,    'aLo' : 1,      'aHi' : 1000},
                 {'cntrSize' : 8,    'aLo' : 10,     'aHi' : 120},
                 {'cntrSize' : 9,    'aLo' : 10,     'aHi' : 10000},
                 {'cntrSize' : 10,   'aLo' : 10,     'aHi' : 10000},
                 {'cntrSize' : 11,   'aLo' : 100,    'aHi' : 10000},
                 {'cntrSize' : 12,   'aLo' : 15,    'aHi' :  100000},
                 {'cntrSize' : 13,   'aLo' : 100,    'aHi' : 100000},
                 {'cntrSize' : 14,   'aLo' : 300,    'aHi' : 100000},
                 {'cntrSize' : 15,   'aLo' : 1000,   'aHi' : 100000},
                 {'cntrSize' : 16,   'aLo' : 1000,   'aHi' : 1000000},
                 ]

class CntrMaster (object):
    """
    Generate, check and parse counters
    """

    # print the details of the counter in a convenient way
    printCntrLine       = lambda self, cntr, expVec, expVal, mantVec, mantVal, cntrVal : print ('expVec={}, expVal={}, mantVec={}, mantVal={}, offset={}, val={}'
                                                                                               .format (expVec, expVal, mantVec, mantVal, self.offsetOfExpVal[expVal], cntrVal))    
    # increment a binary vector, regardless the partition to mantissa, exponent etc.
    # E.g., given a binary vec "00111", this func' will return "01000"  
    incBinVec = lambda self, vec, delta=1 : np.binary_repr (int(vec, base=2)+delta, len(vec)) 

    # Given the cntr's integer value, returns the value it represents 
    cntrInt2num = lambda self, cntrInt : int(1) if (cntrInt==1) else self.a *( (1+1/self.a)**cntrInt - 1)

    # return the maximum value representable by a counter of this size and 'a' parameter
    calcCntrMaxVal  = lambda self : self.cntrInt2num ((1 << self.cntrSize)-1)

    # Given the cntr's vector, returns the it represents value
    cntr2num = lambda self, cntr : self.cntrInt2num (int (cntr, base=2))

    # Generates a strings that details the counter's settings (param vals).    
    genSettingsStr = lambda self : 'Morris_n{}_a{:.2f}' .format (self.cntrSize, self.a)
    
    
    def estimateAloByMaxVal (self, maxVal):
        # return math.log (cntrMaxVal)/math.log (1.001)
        return pow (2^v/m , 1/(maxVal-1))
    
    
    def estimateAGivenCntrSize (self):
        """
        fill a table that, given the cntrSize, estimate Morris counter's "a" parameter to search around for optimizing it.
        Without this function, performing a binary search for 'a' may result in overflow.
        """
        for self.cntrSize in range (5, 8):
            for self.a in [10**i for i in range(3)]:
                CntrMaxVal = self.calcCntrMaxVal ()
                print (f'cntrSize={self.cntrSize}, a={self.a}, CntrMaxVal={CntrMaxVal}')
    
    
    def calcAByMaxVal (self, targetMaxVal, aLo=10, aHi=1000, delta=0.001):
        """
        Given a target maximum countable value, return the maximal 'a' parameter that reaches this value, 
        for the current counter's size.
        the 'a' value determines both the counting range and the expected error: a higher 'a' value decreases the 
        counting range and the estimated error.
        The 'a' value is found by means of a binary search
        Inputs:   
        * aLo - initial lower val for the binary search
        * aHi - initial higher val for the binary search
        * delta = minimum difference (aHi-aLo); when reached - break the binary search.
        """
               
        aSearchRange = [item for item in aSearchRanges if item['cntrSize']==self.cntrSize]
        if len(aSearchRange)==0:
            print ('Sorry, but the requested cntrSize {self.cntrSize} is currently not supported by Morris Counter')
            return
        aLo, aHi = aSearchRange[0]['aLo'], aSearchRange[0]['aHi']
        self.a = aLo
                
        if (self.calcCntrMaxVal() < targetMaxVal):
            print ('cannot reach maxVal={} even with lowest a, aLo={}. Skipping binary search' .format (targetMaxVal, aLo))
            return

        while (aHi - aLo > delta):
            self.a = (aLo + aHi)/2
            maxVal = self.calcCntrMaxVal ()
            if (maxVal==targetMaxVal): # found exact match
                break
            if (maxVal < targetMaxVal): # can't reach maxVal with this a --> need smaller a value
                aHi = self.a
            else: # maxVal > targetMaxVal --> reached the maximum value - try to increase a, to find a tight value.
                aLo = self.a
        if (self.calcCntrMaxVal() < targetMaxVal): # due to quantization, self.a is still too large
            self.a -= delta
        return self.a 
            

    def num2cntr (self, targetVal):
        """
        given a target value, find the closest counters to this targetVal from below and from above.
        Output:
        - A list of dictionaries, where, at each entry, 'cntrVec' is the binary counter, 'val' is its integer value.
        - If an exact match was found (the exact targetVal can be represented), the list contains a single dict entry: the cntr representing this targetVal. 
        - If targetVal <= 0, the list has a single dict entry: the cntr representing 0 
        - If targetVal > maxVal that this cntr can represent, the list has a single dict entry: the cntr repesenting maxVal
        - Else, 
            The first entry in the list is the dict of the max cntr value that is < targetVal.
            The second entry is the dict of min cntr val that is > targetVal.
        """
        if (targetVal > self.cntrMaxVal):
            if (settings.VERBOSE_NOTE in self.verbose):
                print ('Note: the requested cntr value {} is above the max feasible cntr for this configuration' .format(targetVal))
            return [{'cntrVec' : self.cntrMaxVec, 'val' : self.cntrMaxVal}]
        if (targetVal < 0):
            print ('Note: the requested cntr value {} is negative' .format (targetVal))
            return [{'cntrVec' : self.cntrZeroVec, 'val' : 0}]
        
        cntrLoInt = int (math.floor (math.log (1 + targetVal/self.a) * self.num2cntrNormFactor)) # lower-bound on the cntr
        if (self.cntrInt2num(cntrLoInt)==targetVal): # the cntr accurately represents the target value
            return [{'cntrVec' : np.binary_repr(cntrLoInt, self.cntrSize), 'val' : targetVal}]
        
        # now we know that the cntr doesn't accurately represent the target value
        cntrHiInt = cntrLoInt+1
        return [{'cntrVec' : np.binary_repr(cntrLoInt, self.cntrSize), 'val' : self.cntrInt2num(cntrLoInt)},
                {'cntrVec' : np.binary_repr(cntrHiInt, self.cntrSize), 'val' : self.cntrInt2num(cntrHiInt)}]
        

    def __init__ (self, 
                  cntrSize  = 4, # num of bits in each counter. 
                  numCntrs  = 1, # number of counters in the array. 
                  a         = None, # the 'a' parameter that determines the counter's accuracy. 
                  cntrMaxVal= None, 
                  verbose   = [], # determines which outputs would be written to .log/.res/.pcl/debug files, as detailed in settings.py. 
                  estimateAGivenCntrSize=False # When True, only print-out to the screen estimated values of the 'a' parameter to search in, for each counter size - and then exit
                  ):
        
        """
        Initialize an array of cntrSize Morris counters at the given mode. The cntrs are initialized to 0.
        """
        
        if estimateAGivenCntrSize:
            self.estimateAGivenCntrSize ()
            exit ()
        
        if (cntrSize<3):
            print ('error: cntrSize requested is {}. However, cntrSize should be at least 3.' .format (cntrSize))
            exit ()
        self.cntrSize    = int(cntrSize)
        self.numCntrs    = int(numCntrs)
        self.verbose     = verbose
        self.cntrZeroVec = '0' * self.cntrSize
        self.cntrs       = [self.cntrZeroVec for i in range (self.numCntrs)]
        self.cntrMaxVec  = '1' * self.cntrSize
        if (a==None):
            if (cntrMaxVal==None):
                settings.error ('error: the input arguments should include either delta or cntrMaxVal')                
            self.cntrMaxVal = cntrMaxVal
            self.calcAByMaxVal(self.cntrMaxVal)
        else: 
            self.a       = a
        self.cntrZero    = 0
        self.cntrMaxVal  = self.calcCntrMaxVal () #self.cntrInt2num (2**self.cntrSize-1)        
        self.num2cntrNormFactor = 1 / math.log (1 + 1/self.a)
        
    def rstCntr (self, cntrIdx=0):
        """
        """
        self.cntrs[cntrIdx] = self.cntrZeroVec

    def queryCntr (self, cntrIdx=0):
        """
        Query a cntr.
        Input: 
        cntrIdx - the counter's index. 
        Output:
        cntrDic: a dictionary, where: 
            - cntrDict['cntrVec'] is the counter's binary representation; cntrDict['val'] is its value.        
        """
        settings.checkCntrIdx (cntrIdx=cntrIdx, numCntrs=self.numCntrs, cntrType='Morris')
        return {'cntrVec' : self.cntrs[cntrIdx], 'val' : self.cntr2num(self.cntrs[cntrIdx])}    
        
    def incCntrBy1GetVal (self, 
                    cntrIdx  = 0, # idx of the concrete counter to increment in the array
                    forceInc = False) -> dict: # If forceInc==True, increment the counter. Else, inc the counter w.p. corresponding to the next counted value.
        """
        Increment the counter to the closest higher value        
        """
        cntrDict = self.incCntr ()
        return cntrDict['val']
        settings.error ('This function is not debugged yet. Please check it carefully before using.')
        
        if not(forceInc):
            settings.error ('Sorry. Morris.incCntrBy1() is currently supports only for forceInc==True.')
        
        cntrInt = int (self.cntrs[cntrIdx], 2)
        self.cntrs[cntrIdx] = np.binary_repr (cntrInt+1, cntrSize)
        self.cntrs[cntrIdx]    
        # return {'cntrVec' : self.cntrs[cntrIdx], 'val' : self.cntr2num(self.cntrs[cntrIdx])}    
        
    def incCntr (self, cntrIdx=0, factor=1, verbose=[], mult=False):
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
        settings.checkCntrIdx (cntrIdx=cntrIdx, numCntrs=self.numCntrs, cntrType='Morris')
        targetVal = (self.cntr2num (self.cntrs[cntrIdx]) * factor) if mult else (self.cntr2num (self.cntrs[cntrIdx]) + factor)
        optionalModifiedCntr = self.num2cntr (targetVal)
        if (settings.VERBOSE_DETAILS in verbose): 
            if (len(optionalModifiedCntr)==1):
                print ('targetVal={}, cntrLo==cntrHi={}' .format (targetVal, optionalModifiedCntr))
            else:
                print ('targetVal={}, cntrLoVec={}, cntrLoVal={:.2f}\n  cntrHiVec={}, cntrHiVal={:.2f}' .format 
                       (targetVal, optionalModifiedCntr[0]['cntrVec'], optionalModifiedCntr[0]['val'], optionalModifiedCntr[1]['cntrVec'], optionalModifiedCntr[1]['val']))
        if (len(optionalModifiedCntr)==1): # there's a single option to modify the cntr -- either because targetVal is accurately represented, or because it's > maxVal, or < 0.
            self.cntrs[cntrIdx] = optionalModifiedCntr[0]['cntrVec']
        else:
            probOfFurtherInc = float (targetVal - optionalModifiedCntr[0]['val']) / float (optionalModifiedCntr[1]['val'] - optionalModifiedCntr[0]['val'])
            self.cntrs[cntrIdx] = optionalModifiedCntr[1]['cntrVec'] if (random.random() < probOfFurtherInc) else optionalModifiedCntr[0]['cntrVec']
        return {'cntrVec' : self.cntrs[cntrIdx], 'val' : self.cntr2num(self.cntrs[cntrIdx])}    
    
def printAllVals (cntrSize=4, a=None, cntrMaxVal=None, verbose=[]):
    """
    Loop over all the binary combinations of the given counter size. 
    For each combination, print to file the respective counter, and its value. 
    The prints are sorted in an increasing order of values.
    """
    print ('running Morris.printAllVals')
    myCntrMaster = CntrMaster(cntrSize=cntrSize, a=a, cntrMaxVal=cntrMaxVal, verbose=verbose)
    listOfVals = []
    for i in range (1 << cntrSize):
        cntr = np.binary_repr(i, cntrSize) 
        val = myCntrMaster.cntr2num(cntr)
        listOfVals.append ({'cntrVec' : cntr, 'val' : val})
    listOfVals = sorted (listOfVals, key=lambda item : item['val'])

    if (settings.VERBOSE_RES in verbose):
        outputFile    = open ('../res/{}.res' .format (myCntrMaster.genSettingsStr()), 'w')
        for item in listOfVals:
            printf (outputFile, '{}={:.0f}\n' .format (item['cntrVec'], item['val']))


def printAllCntrMaxVals (cntrSizes=[], verbose=[settings.VERBOSE_RES]):
    """
    print the maximum value a cntr reach for several "configurations" -- namely, all combinations of cntrSize and hyperSize. 
    """

    if (settings.VERBOSE_RES in verbose):
        outputFile    = open ('../res/cntrVals.txt', 'a')
    return


# printAllVals (cntrSize=8, a=None, cntrMaxVal=130048, verbose=[settings.VERBOSE_RES]) #1488888
