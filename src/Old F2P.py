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
    genSettingsStr = lambda self : '{}_n{}_h{}' .format (self.mode, self.cntrSize, (self.hyperSize if (self.mode=='F2P') else self.hyperMaxSize))
    
    # returns the value of a number given its offset, exp and mant
    valOf = lambda self, offset, mantVal, expVal : offset + mantVal*2**expVal
    
    # Given an exponent E, calculate the exponent range to which this exponent belongs
    calc_rangeOfExpVal  = lambda self, expVal: max ([j for j in range (len(self.expRange)) if self.expRange[j]<=expVal]) if expVal>0 else 0
    
    # Calculate the maximum feasible hyper-exp size
    calcHyperMaxSize    = lambda self : math.floor((self.cntrSize-1)/2)

    # print the details of the counter in a convenient way
    printCntrLine       = lambda self, cntr, expVec, expVal, mantVec, mantVal, cntrVal : print ('hyperVec={}, expVec={}, bias={}, expVec={}, mantVec={}, mantVal={} \nmantMinSize={}, offset={}, val={}'
                                                                                       .format (cntr[0:self.hyperSize], expVec, self.bias, mantVec, mantVal, self.mantMinSize, self.offsetOfExpVal[expVal], cntrVal))
    

    # Given the values of the mantissa and the exponent, returns the binary cntr representing them - when the mode is F3P.
    mantNexpVals2cntr   = lambda self, mantVal, expVal: self.mantNexpVals2cntrF2P (mantVal, expVal) if (self.mode=='F2P') else self.mantNexpVals2cntrF3P (mantVal, expVal)  

    # Given the vector of the exponent, calculate the value it represents 
    expVec2expVal       = lambda self, expVec, expSize : self.biasOfExpSize[expSize] - (int (expVec, base=2) if expSize>0 else 0)   
    
    # Given the value of the exponent, return the exponent vector representing this value 
    expVal2expVec       = lambda self, expVal, expSize : np.binary_repr(num=int(self.biasOfExpSize[int(expSize)]) - expVal, width=expSize) if expSize>0 else ""   

    # Returns the maximum value that may be represented by this cntr. 
    calcCntrMaxVal      = lambda self : self.calcCntrMaxValF2P() if (self.mode=='F2P') else self.calcCntrMaxValF3P()
     
    # # Returns the maximum value of the counter with its current params 
    # cntrMaxVal          = lambda self : self.cntrMaxVal
    #
    # # Returns the counter that reaches the max value  
    # cntrMaxVec          = lambda self : self.cntrMaxVec
     
    def calcProbOfInc1F2P (self):
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

    def mantNexpVals2cntrF2P (self, mantVal, expVal):
        """
        Given the values of the mantissa and the exponent, returns the binary cntr representing them - when the mode is F2P.
        """

        mantSize = self.mantSizeOfExpVal[expVal]
        expSize  = self.cntrSize - self.hyperSize - mantSize
        return np.binary_repr (num=expSize, width=self.hyperSize) + self.expVal2expVec(expVal=expVal, expSize=expSize) + np.binary_repr (num=mantVal, width=mantSize)
    
    def mantNexpVals2cntrF3P (self, mantVal, expVal):
        """
        Given the values of the mantissa and the exponent, returns the binary cntr representing them - when the mode is F3P.
        """
        
        hyperSize = self.hyperMaxSize - self.rangeOfExpVal[expVal]
        expSize   = hyperSize 
        
        return '1' * hyperSize + \
            ('0' if hyperSize < self.hyperMaxSize else '') + \
            self.expVal2expVec(expVal=expVal, expSize=expSize) + \
            np.binary_repr (num=mantVal, width=self.mantSizeOfExpVal[expVal]) #mantissa

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
        if (self.mode=='F2P'):
            self.mantSizeOfExpVal = np.array ([self.mantMinSize + self.rangeOfExpVal[expVal]     for expVal in range (self.bias+1)])
        else:
            self.mantSizeOfExpVal = np.array ([self.mantMinSize + 2*self.rangeOfExpVal[expVal]-1 for expVal in range (self.bias+1)])
            self.mantSizeOfExpVal[:int(self.expRange[1])] = self.mantMinSize 
            
    def calcParams (self):
        """
        Calc the basics param, which are depended upon the counter size, and the hyper-exp' size.
        """
        self.mantMinSize = self.cntrSize - self.hyperMaxSize - self.expMaxSize 
        if (self.mantMinSize<1):
            print ('cntrSize={} and hyperSize={} implies min mantissa size={}. Mantissa size should be at least 1. Please use a smaller hyperSize' .format(
                    self.cntrSize, self.hyperSize, self.mantMinSize))
            return False
        self.bias        = sum ([2**i for i in range (1, self.expMaxSize+1)])
        self.biasOfExpSize = np.ones (self.expMaxSize+1) #self.biasOfExpSize[j] will hold the bias to be added when the exp size is j
        for j in range (self.expMaxSize+1):
            self.biasOfExpSize[j] = self.bias - sum ([2**i for i in range(j)])
        self.calcOffsets ()
        return True
   
    def __init__ (self, cntrSize=8, hyperSize=1, hyperMaxSize=None, mode='F2P', numCntrs=1, verbose=[]):
        
        """
        Initialize an array of cntrSize counters at the given mode. The cntrs are initialized to 0.
        Inputs:
        cntrSize  - num of bits in each counter.
        hyperSize - size of the hyper-exp field, in bits. Relevant only for F2P counters. 
        hyperMaxSize - maximal size of the hyper-exp field, in bits. Relevant only for F3P counters.
        mode - either 'F2P', or 'F3P'.
        numCntrs - number of counters in the array.
        verbose - can be either:
            settings.VERBOSE_COUT_CNTRLINE - print to stdout details about the concrete counter and its fields.
            settings.VERBOSE_DEBUG         - perform checks and debug operations during the run. 
            settings.VERBOSE_RES           - print output to a .res file in the directory ../res
            settings.VERBOSE_PCL           = print output to a .pcl file in the directory ../res/pcl_files
            settings.VERBOSE_DETAILS       = print to stdout details about the counter
            settings.VERBOSE_NOTE          = print to stdout notes, e.g. when the target cntr value is above its max or below its min.
        """
        
        self.isFeasible = True  # will be False in case of wrong initialization parameters
        if (cntrSize<3):
            print ('error: cntrSize requested is {}. However, cntrSize should be at least 3.' .format (cntrSize))
            exit ()
        self.cntrSize   = int(cntrSize)
        self.numCntrs   = numCntrs
        self.mode       = mode
        self.verbose    = verbose
        if (self.mode=='F2P'):
            if (not (self.setHyperSizeF2P (hyperSize))):
                self.isFeasible = False  
                return                
        elif (self.mode=='F3P'):
            if (not (self.setHyperMaxSize (hyperMaxSize))):
                self.isFeasible = False  
                return
        else:
            print ('error: mode {} is not supported' .format (self.mode)) 
            exit ()
        if (not self.calcParams()): # parameters couldn't be calculated, e.g. due to wrong given combination of cntrSize and hyperSize
            self.isFeasible = False  
            return
        self.calcCntrMaxVal ()
        if self.mode=='F2P':
            self.calcProbOfInc1F2P ()
        self.rstAllCntrs ()
        
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

        if (self.mode=='F2P'):
            return self.cntr2numF2P (cntr, hyperSize) 
        else:
            return self.cntr2numF3P (cntr, hyperMaxSize)

    def calcNprintCntr (self, cntr, expVec, expSize, mantVec):
        """
        Perform the final calculation (which are common for F2P, F3P modes); calculate the counter; and print the res (if requested by the user's verbose).
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
    
    def setHyperSizeF2P (self, hyperSize):
        """
        Sets the size of the hyper-exponent field in F2P counters as follows.
        - Check whether the hyper-exponent field size is feasible.
        - If yes - assign the relevant "self" fields (exponent's field max-size). return True
        - If not - print an error msg and return False
        """
        if (hyperSize<1 or hyperSize>self.cntrSize-2):
            print ('Requested hyperSize {} is not feasible for counter size {}' .format (hyperSize, self.cntrSize))
            return False
        self.hyperSize     = hyperSize
        self.hyperMaxSize  = hyperSize
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
        
    def cntr2numF2P (self, cntr, hyperSize):
        """
        Convert an F2P counter, given as a binary vector (e.g., "11110"), to an integer num.
        Inputs:
        cntr - the counter, given as a binary vector. E.g., "0011"
        hyperSize - size of the hyper-exponent field.
        """
        if (self.hyperSize!=None):
            self.updateSelfHyperSize (hyperSize) # if a new hyperSize was given, override the previous self.hyperSize and update the relevant params
        
        self.hyperVec = cntr [0:self.hyperSize] 
        expSize  = int(self.hyperVec,base=2) 
        return self.calcNprintCntr (cntr=cntr, expVec = cntr[self.hyperSize:self.hyperSize+expSize], expSize=expSize, mantVec=cntr[self.hyperSize+expSize:])

    def queryCntr (self, cntrIdx=0):
        """
        Query a cntr.
        Input: 
        cntrIdx - the counter's index. 
        Output:
        cntrDic: a dictionary, where: 
            - cntrDict['cntrVec'] is the counter's binary representation; cntrDict['val'] is its value.        
        """
        settings.checkCntrIdx (cntrIdx=cntrIdx, numCntrs=self.numCntrs, cntrType=self.mode)        
        
        return {'cntrVec' : self.cntrs[cntrIdx], 'val' : self.cntr2num(self.cntrs[cntrIdx])}    
        
    def incCntrBy1 (self, 
                    cntrIdx  = 0, # idx of the concrete counter to increment in the array
                    forceInc = False) -> dict: # If forceInc==True, increment the counter. Else, inc the counter w.p. corresponding to the next counted value.
        """
        Increment the counter to the closest higher value        
        """
        
        if (self.mode=='F3P'):
            settings.error ('Sorry. incCntrBY1 is not implemented yet for F3P')

        # If the cntr reached its max val, or the randomization decides not to inc, merely return the cur cntr.
        cntr            = self.cntrs[cntrIdx]
        self.hyperVec   = cntr [0:self.hyperSize] 
        expSize         = int(self.hyperVec, base=2)
        expVec          = cntr[self.hyperSize:self.hyperSize+expSize]
        expVal          = int (self.expVec2expVal(expVec, expSize))
        mantVal         = int (cntr[self.hyperSize+expSize:], base=2)
        cntrCurVal      = self.offsetOfExpVal[expVal] + mantVal * (2**expVal)

        if not(forceInc): # check first the case where we don't have to inc the counter 
            if (self.cntrs[cntrIdx]==self.cntrMaxVec or random.random() > self.probOfInc1[int (self.cntrs[cntrIdx], base=2)]): 
                return {'cntrVec' : self.cntrs[cntrIdx], 'val' : cntrCurVal}    

        # now we know that we have to inc. the cntr
        cntrppVal  = cntrCurVal + (1/self.probOfInc1[int (self.cntrs[cntrIdx], base=2)])
        if (mantVal < (1 << self.mantSizeOfExpVal[expVal]) -1): # Can inc. the mantissa without overflowing it  
            self.cntrs[cntrIdx] = self.mantNexpVals2cntr (mantVal+1, expVal)
        else: 
            self.cntrs[cntrIdx] = self.mantNexpVals2cntr (mantVal=0, expVal=expVal+1)
        return {'cntrVec' : self.cntrs[cntrIdx], 'val' : cntrppVal} 
        
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
        
        settings.checkCntrIdx (cntrIdx=cntrIdx, numCntrs=self.numCntrs, cntrType=self.mode)
        self.verbose = verbose
        if not(mult) and self.mode=='F2P' and factor==1:
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
            
    def updateSelfHyperSize (self, hyperSize):
        """
        Sets self.hyperSize, and the relevant fields (self.expMaxSize) to the input hyperSize.
        The function works as follows: 
        If a new hyperSize was given - update self.hyperSize and all the relevant parameters accordingly.
        If no new hyperSize was given, but a hyperSize was stated already upon init - return.
        If no new hyperSize was given, and no hyperSize is already stored in self --> print error msg, saying that hyperSize is not know, and exit. 
        """ 
    
        if (hyperSize==None):
            if (self.hyperSize==None):
                print ('error in F2P mode: hyperSize was nor specified upon init of CntrMaster, neither when calling a function')
                exit ()
            # now we know that hyperSize was already set, and the relevant params were already calculated during __init__.
        else: # hyperSize is not None --> need to re-calculate params accordingly
            self.setHyperSizeF2P(hyperSize)
            self.calcParams ()

          
    def updateHyperMaxSize (self, hyperMaxSize):
        """
        Sets self.hyperMaSize, and the relevant fields (self.expMaxSize) to the input hyperSize.
        If a new hyperMaxSize was given - update self.hyperMaxSize.
        If no new hyperMaxSize was given, but a hyperMaxSize was stated already upon init - return.
        If no new hyperMaxSize was given, and no hyperMaxSize is already stored in self --> set hyperMaxSize to the maximum feasible value.
        If self.hyperMaxSize was now set, set the relevant parameters (e.g., expMaxSize) accordingly. 
        """ 
    
        if (hyperMaxSize==None):
            if (self.hyperMaxSize==None):
                if (settings.VERBOSE_NOTE in self.verbose):
                    print ('note: hyperMaxSize was nor specified upon init of CntrMaster, neither when calling a function, so I am using the largest feasible maxHyperSize')
                self.setHyperMaxSize (hyperMaxSize = self.calcHyperMaxSize ())
                return 
            # No new hyperMaxSize was given, but self.hyperMaxSize and the relevant params were set already upon init --> do nothing  
        else:
            self.setHyperMaxSize (hyperMaxSize) 
            
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
        
    def calcCntrMaxValF3P (self):
        """
        sets self.cntrMaxVal to the maximum value that may be represented by this F3P cntr. 
        """

        self.cntrZeroVec   = np.binary_repr   (2**self.cntrSize - 2**(self.cntrSize-2*self.hyperMaxSize), self.cntrSize) 
        self.cntrMaxVec = np.binary_repr   (2**(self.cntrSize-1)-1, self.cntrSize) # the cntr that reaches the highest value
        self.cntrMaxVal = self.cntr2numF3P (self.cntrMaxVec)

        if (settings.VERBOSE_DEBUG in self.verbose):
            cntrMaxValByFormula = 2**self.bias * (2**(self.cntrSize-1)-1)
            for i in range (self.bias):
                cntrMaxValByFormula += 2**(i + self.mantSizeOfExpVal[i])
        
            if (cntrMaxValByFormula != self.cntrMaxVal):
                print ('error: cntrMaxValByFormula={}, cntrMaxValByCnt={}' .format (cntrMaxValByFormula, self.cntrMaxVal))
                exit ()

    def calcCntrMaxValF2P (self):
        """
        sets self.cntrMaxVal to the maximum value that may be represented by this F2P cntr. 
        """

        self.cntrZeroVec    = np.binary_repr   (2**self.cntrSize - 2**(self.cntrSize-self.hyperSize-self.expMaxSize), self.cntrSize) # the cntr that reaches the lowest value (zero)
        self.cntrMaxVec  = np.binary_repr   (2**(self.cntrSize-self.hyperSize)-1, self.cntrSize) # the cntr that reaches the highest value
        cntrMaxValByCntr = self.cntr2numF2P (self.cntrMaxVec, hyperSize=self.hyperSize)
        self.cntrMaxVal  = cntrMaxValByCntr 

        if (settings.VERBOSE_DEBUG in self.verbose):
            cntrMaxValByFormula = 2**self.bias * (2**(self.cntrSize-self.hyperSize)-1)
            for i in range (self.bias):
                cntrMaxValByFormula += 2**(i + self.mantSizeOfExpVal[i])

            if (cntrMaxValByFormula != cntrMaxValByCntr):
                print ('error: cntrMaxValByFormula={}, cntrMaxValByCnt={}' .format (cntrMaxValByFormula, cntrMaxValByCntr))
                exit ()
        
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
    if (mode=='F2P'):
        for cntrSize in cntrSizeRange:
            for hyperSize in range (1,cntrSize-2) if hyperSizeRange==None else hyperSizeRange:
                myCntrMaster = CntrMaster(mode=mode, cntrSize=cntrSize, hyperSize=hyperSize)
                if (myCntrMaster.isFeasible==False):
                    continue
                if (myCntrMaster.cntrMaxVal < 10**8):
                    printf (outputFile, '{} cntrMaxVal={:.0f}\n' .format (myCntrMaster.genSettingsStr(), myCntrMaster.cntrMaxVal))
                else:
                    printf (outputFile, '{} cntrMaxVal={}\n' .format (myCntrMaster.genSettingsStr(), myCntrMaster.cntrMaxVal))
    elif (mode=='F3P'):
        for cntrSize in cntrSizeRange:
            for hyperMaxSize in range (1,cntrSize-2) if hyperMaxSizeRange==None else hyperMaxSizeRange:
                myCntrMaster = CntrMaster(mode='F3P', cntrSize=cntrSize, hyperMaxSize=hyperMaxSize)
                if (myCntrMaster.isFeasible==False):
                    continue
                if (myCntrMaster.cntrMaxVal < 10**8):
                    printf (outputFile, '{} cntrMaxVal={:.0f}\n' .format (myCntrMaster.genSettingsStr(), myCntrMaster.cntrMaxVal))
                else:
                    printf (outputFile, '{} cntrMaxVal={}\n' .format (myCntrMaster.genSettingsStr(), myCntrMaster.cntrMaxVal))
    else:
        print ('Sorry, mode {} is not supported yet' .format (mode))


# printAllVals (cntrSize=6, hyperSize=1, mode='F2P', verbose=[settings.VERBOSE_RES])