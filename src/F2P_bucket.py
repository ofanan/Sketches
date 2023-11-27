#import itertools
# import time, random, sys, os
# from   pathlib import Path
# from builtins import True False
import math, random, pickle
from printf import printf
import settings
import numpy as np

class CntrMaster (object):
    """
    Generate, check and parse counters
    """

    # Generates a strings that details the counter's settings (param vals).    
    genSettingsStr = lambda self : '{}_n{}_h{}' .format (self.mode, self.cntrSize, self.hyperSize)
    
    # returns the value of a number given its offset, exp and mant
    valOf = lambda self, offset, mantVal, expVal : offset + mantVal*2**expVal
    
    # Given an exponent E, calculate the exponent range to which this exponent belongs
    calc_rangeOfExpVal  = lambda self, expVal: max ([j for j in range (len(self.expRange)) if self.expRange[j]<=expVal]) if expVal>0 else 0
    
    # Calculate the maximum feasible hyper-exp size
    calcHyperMaxSize    = lambda self : math.floor((self.cntrSize-1)/2)

    # print the details of the counter in a convenient way
    printCntrLine       = lambda self, cntr, expVec, expVal, mantVec, mantVal, cntrVal : print ('hyperVec={}, expVec={}, bias={}, expVec={}, mantVec={}, mantVal={} \nmantMinSize={}, offset={}, val={}'
                                                                                       .format (cntr[0:self.hyperSize], expVec, self.bias, mantVec, mantVal, self.mantMinSize, self.offsetOfExpVal[expVal], cntrVal))
    # Given the vector of the exponent, calculate the value it represents 
    expVec2expVal       = lambda self, expVec, expSize : self.biasOfExpSize[expSize] - (int (expVec, base=2) if expSize>0 else 0)   
    
    # Given the value of the exponent, return the exponent vector representing this value 
    expVal2expVec       = lambda self, expVal, expSize : np.binary_repr(num=int(self.biasOfExpSize[int(expSize)]) - expVal, width=expSize) if expSize>0 else ""   

    def calcProbOfInc1 (self):
        """
        Calculate the array self.probOfInc1, which is defined as follows.
        self.probOfInc1[i] = the prob' of incrementing the counter by 1, when the value of the cntr is i.
        This is calculated as: self.probOfInc1[i] = 1/(value_of_the_cntr_if_incremented_by_1 - curCntrVal) 
        """

        self.probOfInc1 = np.ones (2**self.cntrSize)

        if (settings.VERBOSE_RES in self.verbose):
            listOfVals = [] 

        for i in range (2**self.cntrSize):
            cntr            = np.binary_repr(i, self.cntrSize) 
            self.hyperVec   = cntr [0:self.hyperSize] 
            expSize         = int(self.hyperVec,base=2) 
            expVal          = int (self.expVec2expVal (expVec=cntr[self.hyperSize:self.hyperSize+expSize], expSize=expSize))
            mantVal         = int (cntr[self.hyperSize+expSize:], base=2)
            offset          = self.offsetOfExpVal[int(expVal)]
            cntrVal         = self.valOf (offset, mantVal, expVal)
            if (cntrVal==self.cntrMaxVal):
                continue # for cntrMaxVal, the prob' of further inc is the default --> 0

            if (mantVal < (1 << self.mantSizeOfExpVal[expVal]) -1): # can still inc. the mantissa 
                self.probOfInc1[i] = 1/(self.valOf(offset, mantVal+1, expVal) - cntrVal)
            else: # cannot inc. the mantissa --> inc. the exp
                self.probOfInc1[i] = 1/(self.valOf(offset=self.offsetOfExpVal[expVal+1], mantVal=0, expVal=expVal+1) - cntrVal) 
            if (settings.VERBOSE_RES in self.verbose):
                listOfVals.append ({'cntrVec' : cntr, 'val' : cntrVal, 'prob' : self.probOfInc1[i]})
                
        if (settings.VERBOSE_DETAILED_RES in self.verbose):
            outputFile    = open ('../res/{}_probs.res' .format (self.genSettingsStr()), 'w')
            listOfVals = sorted (listOfVals, key=lambda item : item['val'])
            for item in listOfVals:
                printf (outputFile, '{}={:.0f} | prob={}\n' .format (item['cntrVec'], item['val'], item['prob']))
    
    def calcExpRanges (self):
        """
        Calculate the ranges of the exponent (E_0, E_1, ...)
        """
        self.expRange = np.zeros (self.expMaxSize+1)
        for j in range (0, self.expMaxSize):
            self.expRange[j+1] = int (sum ([2**(i) for i in range (self.expMaxSize-j, self.expMaxSize+1)]))

    def mantNexpVals2cntr (self, mantVal, expVal):
        """
        Given the values of the mantissa and the exponent, returns the binary cntr representing them - when the mode is F2P.
        """

        mantSize = self.mantSizeOfExpVal[expVal]
        expSize  = self.cntrSize - self.hyperSize - mantSize
        return np.binary_repr (num=expSize, width=self.hyperSize) + self.expVal2expVec(expVal=expVal, expSize=expSize) + np.binary_repr (num=mantVal, width=mantSize)
    
    def calcOffsets (self):
        """
        Pre-calculate all the offsets to be added to a counter, according to its exponent value.
        self.offsetOfExpVal[e] will hold the offset to be added to the counter's val when the exponent's value is e.
        """
        self.calcExpRanges        ()
        self.calcMantSizeOfExpVal ()
        self.offsetOfExpVal = np.zeros (self.bias+1) #self.offsetOfExpVal[j] will hold the value to be added to the counter when the exponent is j
        for expVal in range (self.bias): # for each potential exponent value
            self.offsetOfExpVal[expVal+1] = self.offsetOfExpVal[expVal] + 2**(expVal+self.mantSizeOfExpVal[expVal])

    def calcMantSizeOfExpVal (self):
        """
        Calculate M(E), namely, the size of the mantissa implied by having each given value of exponent. 
        In particular, this function fills the array
        self.mantSizeOfExpVal, where self.mantSizeOfExpVal[e] is the size of mantissa implied by a given exponent value.
        """
        self.rangeOfExpVal = np.zeros (self.bias+1, dtype='uint16')
        for expVal in range (self.bias+1):
            self.rangeOfExpVal[expVal] = self.calc_rangeOfExpVal(expVal)
        self.mantSizeOfExpVal = np.array ([self.mantMinSize + self.rangeOfExpVal[expVal]     for expVal in range (self.bias+1)])
            
    def calcParams (self):
        """
        Calc the basics param, which are depended upon the counter size, and the hyper-exp' size.
        """
        self.mantMinSize = self.cntrSize - self.hyperMaxSize - self.expMaxSize 
        if (self.mantMinSize<1):
            print (f'cntrSize={self.cntrSize} and hyperSize={self.hyperSize} implies min mantissa size={self.mantMinSize}. Mantissa size should be at least 1. Please use a smaller hyperSize')
            return False
        self.bias        = sum ([2**i for i in range (1, self.expMaxSize+1)])
        self.biasOfExpSize = np.ones (self.expMaxSize+1) #self.biasOfExpSize[j] will hold the bias to be added when the exp size is j
        for j in range (self.expMaxSize+1):
            self.biasOfExpSize[j] = self.bias - sum ([2**i for i in range(j)])
        self.calcOffsets ()
        return True
   
    def __init__ (self, cntrSize=8, hyperSize=1, numCntrs=1, verbose=[]):
        
        """
        Initialize an array of cntrSize counters at the given mode. The cntrs are initialized to 0.
        Inputs:
        cntrSize  - num of bits in each counter.
        hyperSize - size of the hyper-exp field, in bits.  
        numCntrs - number of counters in the array.
        verbose - can be either:
            settings.VERBOSE_COUT_CNTRLINE - print to stdout details about the concrete counter and its fields.
            settings.VERBOSE_DEBUG         - perform checks and debug operations during the run. 
            settings.VERBOSE_RES           - print output to a .res file in the directory ../res
            settings.VERBOSE_PCL           = print output to a .pcl file in the directory ../res/pcl_files
            settings.VERBOSE_DETAILS       = print to stdout details about the counter
            settings.VERBOSE_NOTE          = print to stdout notes, e.g. when the target cntr value is above its max or below its min.
        """
        
        if (cntrSize<3):
            print ('error: cntrSize requested is {}. However, cntrSize should be at least 3.' .format (cntrSize))
            exit ()
        self.cntrSize   = int(cntrSize)
        self.numCntrs   = int(numCntrs)
        self.verbose    = verbose
        if hyperSize==0:
            self.hyperSize = 0
            self.cntrMaxVal  = (1 << self.cntrSize) - 1 
            if self.cntrSize<=8:
                self.cntrs      = np.zeros(self.numCntrs, dtype='uint8') 
            elif self.cntrSize<=16:
                self.cntrs      = np.zeros(self.numCntrs, dtype='uint16') 
            else:
                self.cntrs      = np.zeros(self.numCntrs, dtype='uint32')             
            return            
        if (not (self.setHyperSize (hyperSize))):
            self.isFeasible = False  
            return                
        if (not self.calcParams()): # parameters couldn't be calculated, e.g. due to wrong given combination of cntrSize and hyperSize
            self.isFeasible = False  
            return
        self.calcCntrMaxVal ()
        if self.hyperSize>0:
            self.calcProbOfInc1 ()
        self.rstAllCntrs ()
    
    def incHyperExpSize (self):
        """
        Increment the size of the hyper-exponent field by 1.
        In particular:
        - Edit all counters according to the new format.  
        """
        self.cntrSize  += 1
        self.hyperSize += 1
        if hyperSize>self.cntrSize-2:
            settings.error (f'Requested hyperSize {self.hyperSize} is not feasible for counter size {self.cntrSize}')
            return False
        self.expMaxSize    = 2**(self.hyperSize)-1 # the maximum value that can be represented by self.hyperSize bits, using standard binary representation. 
        if (self.hyperSize + self.expMaxSize > self.cntrSize-1):
            print ('Requested hyperSize {} is not feasible for counter size {}' .format (hyperSize, self.cntrSize))
            return False
        
        self.calcParams() # parameters couldn't be calculated, e.g. due to wrong given combination of cntrSize and hyperSize
        self.calcCntrMaxVal ()
        self.calcProbOfInc1 ()
        self.cntrs = [self.cntrZeroVec for _ in range (self.numCntrs)]
        
    def rstAllCntrs (self):
        """
        """
        self.cntrs = [self.cntrZeroVec for _ in range (self.numCntrs)]
        
    def rstCntr (self, cntrIdx=0):
        """
        """
        self.cntrs[cntrIdx] = self.cntrZeroVec
        
        
    def cntr2num (self, cntr, hyperSize=None, hyperMaxSize=None, verbose=[]):
        """
        Convert a counter, given as a binary vector (e.g., "11110"), to an integer num.
        """
        if (verbose!=[]):
            self.verbose = verbose
        if (len(cntr) != self.cntrSize): # if the cntr's size differs from the default, we have to update the basic params
            print ('the size of the given counter is {} while CntrMaster was initialized with cntrSize={}' .format (len(cntr), self.cntrSize))
            exit ()        

        return self.cntr2num (cntr, hyperSize) 

    def calcNprintCntr (self, cntr, expVec, expSize, mantVec):
        """
        Perform the final calculation; calculate the counter; and print the res (if requested by the user's verbose).
        Returns the value of the cntr (as int). 
        """
        expVal   = self.expVec2expVal(expVec, expSize) 
        if (settings.VERBOSE_DEBUG in self.verbose):
            if (expVec != self.expVal2expVec(expVal, expSize=expSize)):   
                print ('error: expVec={}, expVal={}, expSize={}, Back to expVec={}' .format (expVec, expVal, expSize, self.expVal2expVec(expVal, expSize)))
                exit ()
        mantVal  = int (mantVec, base=2)
        cntrVal  = self.offsetOfExpVal[int(expVal)] + mantVal * (2**expVal)
        if (settings.VERBOSE_COUT_CNTRLINE in self.verbose):
            self.printCntrLine (cntr=cntr, expVec=expVec, expVal=expVal, mantVal=mantVal, cntrVal=cntrVal)
        return cntrVal
    
    def setHyperSize (self, hyperSize):
        """
        Sets the size of the hyper-exponent field in F2P counters as follows.
        - Check whether the hyper-exponent field size is feasible.
        - If yes - assign the relevant "self" fields (exponent's field max-size). return True
        - If not - print an error msg and return False
        """
        if (hyperSize<0 or hyperSize>self.cntrSize-2):
            print ('Requested hyperSize {} is not feasible for counter size {}' .format (hyperSize, self.cntrSize))
            return False
        self.hyperSize     = hyperSize
        self.expMaxSize    = 2**(self.hyperSize)-1 # the maximum value that can be represented by self.hyperSize bits, using standard binary representation. 
        if (self.hyperSize + self.expMaxSize > self.cntrSize-1):
            print ('Requested hyperSize {} is not feasible for counter size {}' .format (hyperSize, self.cntrSize))
            return False
        return True

    def setHyperMaxSize (self, hyperMaxSize):
        """
        Sets the maximal size of the hyper-exponent field in F3P counters as follows.
        - Check whether the hyper-exponent field size is feasible.
        - If yes - assign the relevant "self" fields (exponent's field max-size, which is identical to hyperMaxSize). Return True
        - If not - print an error msg and return False
        """
        hyperMaxSize = self.calcHyperMaxSize() if (hyperMaxSize==None) else hyperMaxSize
        if (2*hyperMaxSize > self.cntrSize-1):
            print ('error: requested hyperSize {} is not feasible for counter size {}' .format (hyperMaxSize, self.cntrSize))
            return False
        self.hyperMaxSize  = hyperMaxSize
        self.expMaxSize    = self.hyperMaxSize
        return True  

    
    def cntr2numF3P (self, cntr, hyperMaxSize=None):
        """
        Convert an F3P counter, given as a binary vector (e.g., "11110"), to an integer num.
        Inputs:
        cntr - the counter, given as a binary vector. E.g., "0011"
        hyperMaxSize - maximum size of the hyper-exp field.
        Output:
        the integer value of the given cntr.    
        """
        self.updateHyperMaxSize (hyperMaxSize)
        
        # Extract the hyper-exponent field, and value
        self.hyperSize = settings.idxOfLeftmostZero (ar=cntr, maxIdx=self.hyperMaxSize)         
        expSize      = self.hyperSize
        if (self.hyperSize < self.hyperMaxSize): # if the # of trailing max < hyperMaxSize, the cntr must have a a delimiter '0'
            expVecBegin  = self.hyperSize+1
        else:
            expVecBegin  = self.hyperMaxSize

        return self.calcNprintCntr (cntr=cntr, expVec = cntr[expVecBegin : expVecBegin+expSize], expSize=expSize, mantVec=cntr[expVecBegin+expSize:])
        
    def cntr2num (self, cntr):
        """
        Convert an F2P counter, given as a binary vector (e.g., "11110"), to an integer num.
        Inputs:
        cntr - the counter, given as a binary vector. E.g., "0011"
        """
        self.hyperVec = cntr [0:self.hyperSize] 
        expSize  = int(self.hyperVec,base=2) 
        return self.calcNprintCntr (cntr=cntr, expVec = cntr[self.hyperSize:self.hyperSize+expSize], expSize=expSize, mantVec=cntr[self.hyperSize+expSize:])

    def queryCntrGetVal (self, cntrIdx=0):
        """
        Query a cntr.
        Input: 
        cntrIdx - the counter's index. 
        Output:
        cntrDic: a dictionary, where: 
            - cntrDict['cntrVec'] is the counter's binary representation; cntrDict['val'] is its value.        
        """
        return self.cntr2num(self.cntrs[cntrIdx])    

    def incCntrBy1GetVal (self, 
                    cntrIdx  = 0): # idx of the concrete counter to increment in the array
        """
        Increment the counter to the closest higher value.
        If the counter is already the maximal value, do nothing.
        Else, increment the counter with prob' 1/(newValue-curValue).
        Return:
        - the value after increment.
        - True iff the value was incremented.  
        """

        if self.hyperSize==0: # case where this is merely a standard integer counter
            if self.cntrs[cntrIdx]==self.cntrMaxVal:
                return self.cntrs[cntrIdx], False
            self.cntrs[cntrIdx] += 1
            return self.cntrs[cntrIdx], True
        # If the cntr reached its max val, or the randomization decides not to inc, merely return the cur cntr.
        cntr            = self.cntrs[cntrIdx]
        self.hyperVec   = cntr [0:self.hyperSize] 
        expSize         = int(self.hyperVec, base=2)
        expVec          = cntr[self.hyperSize:self.hyperSize+expSize]
        expVal          = int (self.expVec2expVal(expVec, expSize))
        mantVal         = int (cntr[self.hyperSize+expSize:], base=2)
        cntrCurVal      = self.offsetOfExpVal[expVal] + mantVal * (2**expVal)

        if (self.cntrs[cntrIdx]==self.cntrMaxVec or random.random() > self.probOfInc1[int (self.cntrs[cntrIdx], base=2)]): 
            return cntrCurVal, False

        # now we know that we have to inc. the cntr
        cntrppVal  = cntrCurVal + (1/self.probOfInc1[int (self.cntrs[cntrIdx], base=2)])
        if (mantVal < (1 << self.mantSizeOfExpVal[expVal]) -1): # Can inc. the mantissa without overflowing it  
            self.cntrs[cntrIdx] = self.mantNexpVals2cntr (mantVal+1, expVal)
        else: 
            self.cntrs[cntrIdx] = self.mantNexpVals2cntr (mantVal=0, expVal=expVal+1)
        return cntrppVal, True
        
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
        settings.checkCntrIdx (cntrIdx=cntrIdx, numCntrs=self.numCntrs, cntrType=self.mode)
        self.verbose = verbose
        if not(mult) and factor==1:
            return self.incCntrBy1(cntrIdx=cntrIdx)     
        targetVal = (self.cntr2num (self.cntrs[cntrIdx]) * factor) if mult else (self.cntr2num (self.cntrs[cntrIdx]) + factor)
        optionalModifiedCntr = self.num2cntr (targetVal)
        if (len(optionalModifiedCntr)==1): # there's a single option to modify the cntr -- either because targetVal is accurately represented, or because it's > maxVal, or < 0.
            self.cntrs[cntrIdx] = optionalModifiedCntr[0]['cntrVec']
        else:
            probOfFurtherInc = float (targetVal - optionalModifiedCntr[0]['val']) / float (optionalModifiedCntr[1]['val'] - optionalModifiedCntr[0]['val'])
            if (random.random() < probOfFurtherInc): 
                self.cntrs[cntrIdx] = optionalModifiedCntr[1]['cntrVec'] 
            else:
                self.cntrs[cntrIdx] = optionalModifiedCntr[0]['cntrVec']
        return {'cntrVec' : self.cntrs[cntrIdx], 'val' : self.cntr2num(self.cntrs[cntrIdx])}    
            
            
    def num2cntr (self, targetVal, verbose=None):
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
        if (verbose!=None): #if a new verbose was given, it overrides the current verbose
            self.verbose = verbose
        if (targetVal > self.cntrMaxVal):
            if (settings.VERBOSE_NOTE in self.verbose):
                print ('Note: the requested cntr value {} is above the max feasible cntr for this configuration' .format(targetVal))
            return [{'cntrVec' : self.cntrMaxVec, 'val' : self.cntrMaxVal}]
        if (targetVal < 0):
            if (settings.VERBOSE_NOTE in self.verbose):
                print ('Note: the requested cntr value {} is negative' .format (targetVal))
            return [{'cntrVec' : self.cntrZeroVec, 'val' : 0}]
        
        offset  = max ([offset for offset in self.offsetOfExpVal if offset<=targetVal])
        expVal  = list(self.offsetOfExpVal).index(offset)
        mantVal = math.floor (float(targetVal-offset)/float(1 << expVal))
        cntr    = self.mantNexpVals2cntr (mantVal, expVal)
        cntrVal = self.valOf(offset, mantVal, expVal)
        if (settings.VERBOSE_DEBUG in self.verbose):
            numVal  = self.cntr2num(cntr=cntr, hyperSize=self.hyperSize)
            if (cntrVal != numVal):
                print ('error in num2cntr: cntrVal={}, but the val of the generated cntr={}' .format (cntrVal, self.cntr2num(cntr)))
                exit ()
        if (cntrVal==targetVal): # found a cntr that accurately represents the target value
            return [{'cntrVec' : cntr, 'val' : cntrVal}]

        # now we know that the counter found is < the target value
        if (mantVal < (1 << self.mantSizeOfExpVal[expVal]) -1): 
            cntrpp    = self.mantNexpVals2cntr (mantVal+1, expVal)
            cntrppVal = self.valOf(offset, mantVal+1, expVal)
        else: 
            cntrpp    = self.mantNexpVals2cntr (mantVal=0, expVal=expVal+1)
            cntrppVal = self.valOf(offset=self.offsetOfExpVal[expVal+1], mantVal=0, expVal=expVal+1)
        return [{'cntrVec' : cntr, 'val' : cntrVal}, {'cntrVec' : cntrpp, 'val' : cntrppVal}]        
        

    def calcCntrMaxVal (self):
        """
        sets self.cntrMaxVal to the maximum value that may be represented by this F2P cntr. 
        """

        self.cntrZeroVec = np.binary_repr   (2**self.cntrSize - 2**(self.cntrSize-self.hyperSize-self.expMaxSize), self.cntrSize) # the cntr that reaches the lowest value (zero)
        self.cntrMaxVec  = np.binary_repr   (2**(self.cntrSize-self.hyperSize)-1, self.cntrSize) # the cntr that reaches the highest value
        self.cntrMaxVal  = self.cntr2num (self.cntrMaxVec, hyperSize=self.hyperSize) 
        
def printAllVals (cntrSize=8, hyperSize=2, hyperMaxSize=2, mode='F3P', verbose=[]):
    """
    Loop over all the binary combinations of the given counter size. 
    For each combination, print to file the respective counter, and its value. 
    The prints are sorted in an increasing order of values.
    """
    print ('running printAllVals. mode={}' .format (mode))
    myCntrMaster = CntrMaster(cntrSize=cntrSize, hyperSize=hyperSize, hyperMaxSize=hyperMaxSize, mode=mode)
    listOfVals = []
    if (mode=='F2P'):
        for i in range (2**cntrSize):
            cntr = np.binary_repr(i, cntrSize) 
            val = myCntrMaster.cntr2num(cntr, hyperSize=hyperSize)
            listOfVals.append ({'cntrVec' : cntr, 'val' : val})
    elif (mode=='F3P'):
        for i in range (2**cntrSize):
            cntr = np.binary_repr(i, cntrSize) 
            val = myCntrMaster.cntr2num(cntr, hyperMaxSize=hyperMaxSize)
            listOfVals.append ({'cntrVec' : cntr, 'val' : val})
    else:
        print ('sorry, mode {} that you chose is not supported yet' .format (mode))
        exit ()
    listOfVals = sorted (listOfVals, key=lambda item : item['val'])
    
    if (settings.VERBOSE_RES in verbose):
        outputFile    = open ('../res/{}.res' .format (myCntrMaster.genSettingsStr()), 'w')
        for item in listOfVals:
            printf (outputFile, '{}={:.0f}\n' .format (item['cntrVec'], item['val']))
    
    if (settings.VERBOSE_PCL in verbose):
        with open('../res/pcl_files/{}.pcl' .format (myCntrMaster.genSettingsStr()), 'wb') as pclOutputFile:
            pickle.dump(listOfVals, pclOutputFile) 
      

def printAllCntrMaxVals (mode = 'F3P', hyperSizeRange=None, hyperMaxSizeRange=None, cntrSizeRange=[], verbose=[settings.VERBOSE_RES]):
    """
    print the maximum value a cntr reach for several "configurations" -- namely, all combinations of cntrSize and hyperSize. 
    """

    if (settings.VERBOSE_RES in verbose):
        outputFile    = open ('../res/cntrMaxVals.txt', 'a')
    for cntrSize in cntrSizeRange:
        for hyperSize in range (1,cntrSize-2) if hyperSizeRange==None else hyperSizeRange:
            myCntrMaster = CntrMaster(mode=mode, cntrSize=cntrSize, hyperSize=hyperSize)
            if (myCntrMaster.isFeasible==False):
                continue
            if (myCntrMaster.cntrMaxVal < 10**8):
                printf (outputFile, '{} cntrMaxVal={:.0f}\n' .format (myCntrMaster.genSettingsStr(), myCntrMaster.cntrMaxVal))
            else:
                printf (outputFile, '{} cntrMaxVal={}\n' .format (myCntrMaster.genSettingsStr(), myCntrMaster.cntrMaxVal))

# printAllVals (cntrSize=6, hyperSize=1, mode='F2P', verbose=[settings.VERBOSE_RES])