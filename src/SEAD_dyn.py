import math, time, random, numpy as np

from printf import printf
import settings, SEAD_stat
from settings import *

class CntrMaster (SEAD_stat.CntrMaster):
    """
    Generate, check and parse counters
    """

    # Generates a strings that details the counter's settings (param vals).    
    genSettingsStr = lambda self : f'SEADdyn_n{self.cntrSize}_e0'
    
    # get the mantissa value in 'stat' mode  
    getMantVal = lambda self, cntrIdx, expSize : int (self.cntrs[cntrIdx][expSize:], base=2)
    
    # Return a range with all the legal combinations for the counter 
    getAllCombinations = lambda self, cntrSize : range (2**cntrSize-2)
    
    def calcParams (self):
        """
        Pre-compute the cntrs' parameters, in case of a dynamic SEAD cntr 
        """
        self.cntrMaxVec = '1' * (self.cntrSize-2) + '0' + '1'
        self.expMaxVal  = self.cntrSize-2
        self.offsetOfExpVal = [expVal * 2**(self.cntrSize-1) for expVal in range (self.expMaxVal+1)]
        self.cntrMaxVal = self.valOf (mantVal=1, expVal=self.expMaxVal)                 
   
    def cntr2num (self, 
                  cntr, # the counter, given as a binary vector (e.g., "11110"). 
                  ):
        """
        Convert a counter, given as a binary vector (e.g., "11110"), to an integer num.
        Output: integer.
        """        
        if (len(cntr) != self.cntrSize): # if the cntr's size differs from the default, we have to update the basic params
            print ('the size of the given counter is {} while CntrMaster was initialized with cntrSize={}.' .format (len(cntr), self.cntrSize))
            print ('Please initialize a cntr with the correct len.')
            exit ()        
        expSize = settings.idxOfLeftmostZero (ar=cntr, maxIdx=self.cntrSize-2)
        expVec  = cntr[:expSize]
        mantVec = cntr[expSize+1:]
        if (settings.VERBOSE_COUT_CNTRLINE in self.verbose):
            mantVal = int (mantVec, base=2)
            cntrVal = self.valOf (expVal=expSize, mantVal=mantVal)
            self.printCntrLine (cntr=cntr, expVec=expVec, expVal=expSize, mantVec=mantVec, mantVal=mantVal, cntrVal=cntrVal)
        return self.valOf (expVal=expSize, mantVal=int (mantVec, base=2))

    def incCntr (self, cntrIdx=0):
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

        if verbose!=None:
            self.verbose = verbose
        if factor==1 and mult==False:
            return self.incCntrBy1GetVal (cntrIdx)
        warning ('In SEAD_dyn.incCntr(). Please carefully check this func before using it.')        
        offset  = max ([offset for offset in self.offsetOfExpVal if offset<=self.targetVal]) # find the maximal offset which is <= targetVal
        expSize = list(self.offsetOfExpVal).index(offset) # expSize is the index of this offset in self.offsetOfExpVal 
        mantVal = math.floor (float(self.targetVal-offset)/float(2**expSize))
        mantSize = self.cntrSize - expSize - 1
        self.cntrs[cntrIdx] = '1' * expSize + '0' + np.binary_repr(mantVal, mantSize)
        cntrVal = self.cntr2num(self.cntrs[cntrIdx])

        if (cntrVal==self.targetVal): # does the cntr accurately represent the target value?
            return {'cntrVec' : self.cntrs[cntrIdx], 'val' : cntrVal} # yep --> return the counter, and its value

        # now we know that the counter found is < the target value
        if (mantVal < 2**mantSize-1): # can we further increment the mantissa w/o o/f?
            cntrpp    = '1' * expSize     + '0' + np.binary_repr (mantVal+1, mantSize)
        else:  # need to increase the exponent (and thus, also its size)
            cntrpp    = '1' * (expSize+1) + '0' * mantSize # need to decrement the mantissa field size.
        cntrppVal = self.cntr2num (cntrpp)

        probOfFurtherInc = float (self.targetVal - cntrVal) / float (cntrppVal - cntrVal)
        if (random.random() < probOfFurtherInc):
            self.cntrs[cntrIdx] = cntrpp
            return {'cntrVec' : self.cntrs[cntrIdx], 'val' : cntrppVal}
        return {'cntrVec' : self.cntrs[cntrIdx], 'val' : cntrVal} 
                
    def incCntrBy1GetVal (self, cntrIdx=0):
        """
        Increase a counter by 1.
        Operation:
        Define cntrVal as the current counter's value. 
        Then, targetValue = cntrVal+1  
        If targetValue > maximum cntr's value, return the a cntr representing the max possible value. 
        If targetValue < 0, return a cntr representing 0.
        If targetValue can be represented correctly by the counter, return the exact representation.
        Else, use probabilistic cntr's modification.
        Return the updated cntr's value.
        """
        cntr = self.cntrs[cntrIdx]
        expSize = settings.idxOfLeftmostZero (ar=cntr, maxIdx=self.cntrSize-2)
        expVec  = cntr[:expSize]
        mantVec = cntr[expSize+1:]
        if (settings.VERBOSE_COUT_CNTRLINE in self.verbose):
            mantVal = int (mantVec, base=2)
            cntrVal = self.valOf (expVal=expSize, mantVal=mantVal)
            self.printCntrLine (cntr=cntr, expVec=expVec, expVal=expSize, mantVec=mantVec, mantVal=mantVal, cntrVal=cntrVal)

        cntrCurVal = self.valOf (expVal=expSize, mantVal=int (mantVec, base=2))
        if cntrCurVal == self.cntrMaxVal:
            return cntrCurVal

        cntrppVal = cntrCurVal + 2**expSize

        if random.random() >= 1/float(cntrppVal-cntrCurVal):
            return cntrCurVal 

        # Need to increment the cntr
        mantVal = self.getMantVal(cntrIdx, expSize=expSize)
        mantSize = self.cntrSize - expSize - 1
        if (mantVal < 2**mantSize-1): # can we further increment the mantissa w/o o/f?
            self.cntrs[cntrIdx] = '1'* expSize    + '0' + np.binary_repr (mantVal+1, mantSize)
        else:  # need to increase the exponent
            self.cntrs[cntrIdx] = '1'*(expSize+1) + '0' * mantSize # a single delimiter '0' between the exponent and the mantissa + a shrinked-by-one reset mantissa field.
        if settings.VERBOSE_LOG_CNTRLINE in self.verbose:
            printf (self.logFIle, f'After inc: cntrVec={self.cntrs[cntrIdx]}, cntrVal={cntrppVal}\n')
        return cntrppVal


def printAllCntrMaxVals (cntrSizes=[], verbose=[settings.VERBOSE_RES]):
    """
    print the maximum value a cntr reach for several "configurations" -- namely, all combinations of cntrSize and hyperSize. 
    """

    if (settings.VERBOSE_RES in verbose):
        outputFile    = open ('../res/cntrMaxVals.txt', 'a')
    for cntrSize in cntrSizes:
        for cntrSize in cntrSizes:
            myCntrMaster = CntrMaster(mode='dyn', cntrSize=cntrSize)
            printf (outputFile, '{} cntrMaxVal={:.0f}\n' .format (myCntrMaster.genSettingsStr (mode='dyn', cntrSize=cntrSize), myCntrMaster.cntrMaxVal))

# myCntrMaster = CntrMaster (cntrSize=8)
# myCntrMaster.getAllVals (verbose=[settings.VERBOSE_RES])
