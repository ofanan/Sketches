#import itertools
# import time, random, sys, os
# from   pathlib import Path
# from builtins import True False
# import pickle
import math, time, random
from printf import printf
import settings
import numpy as np

class CntrMaster (object):
    """
    Generate, check and parse counters
    """

    # Generates a strings that details the counter's settings (param vals).    
    genSettingsStr = lambda self : 'SEAD{}_n{}_e{}' .format ('stat' if self.mode=='stat' else 'dyn', self.cntrSize, self.expSize if self.mode=='stat' else 0)
    
    # print the details of the counter in a convenient way
    printCntrLine       = lambda self, cntr, expVec, expVal, mantVec, mantVal, cntrVal : print ('expVec={}, expVal={}, mantVec={}, mantVal={}, offset={}, val={}'
                                                                                               .format (expVec, expVal, mantVec, mantVal, self.offsetOfExpVal[expVal], cntrVal))    
    # returns the value of a cntr given its exp and mant
    valOf = lambda self, mantVal, expVal : self.offsetOfExpVal[expVal] + mantVal*2**expVal
     
    # increment a binary vector, regardless the partition to mantissa, exponent etc.
    # E.g., given a binary vec "00111", this func' will return "01000"  
    incBinVec = lambda self, vec, delta=1 : np.binary_repr (int(vec, base=2)+delta, len(vec)) 

    # get the exponent vector in 'stat' mode 
    getExpVecStat  = lambda self, cntrIdx : self.cntrs[cntrIdx][:self.expSize]            

    # get the mantisa vector in 'stat' mode 
    getMantVecStat = lambda self, cntrIdx : self.cntrs[cntrIdx][self.expSize:]
                
    # get the exponent value in 'stat' mode  
    getExpValStat  = lambda self, cntrIdx : int (self.getExpVecStat(cntrIdx), base=2)             

    # get the mantissa value in 'stat' mode  
    getMantValStat = lambda self, cntrIdx : int (self.getMantVecStat(cntrIdx), base=2)
    
    def calcOffsets (self):
        """
        Pre-calculate all the offsets to be added to a counter, according to its exponent value:
        self.offsetOfExpVal[e] will hold the offset to be added to the counter's val when the exponent's value is e.
        """
        self.offsetOfExpVal   = np.zeros (self.expMaxVal+1)  
        if (self.mode=='stat'):
            for expVal in range (self.expMaxVal): # for each potential exponent value
                self.offsetOfExpVal[expVal+1] = self.offsetOfExpVal[expVal] + 2**(expVal+self.mantSize)
        else:
            self.offsetOfExpVal = [expVal * 2**(self.cntrSize-1) for expVal in range (self.expMaxVal+1)]
  
    def __init__ (self, cntrSize=4, expSize=2, mode='stat', numCntrs=1, verbose=[]):
        
        """
        Initialize an array of cntrSize counters at the given mode. The cntrs are initialized to 0.
        Inputs:
        cntrSize  - num of bits in each counter.
        expSize - size of the exp field, in bits. Relevant only for static counters. 
        mode - either 'stat', or 'dyn'.
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
        self.cntrSize    = int(cntrSize)
        self.numCntrs    = int(numCntrs)
        self.verbose     = verbose
        self.cntrZeroVec = '0' * self.cntrSize
        self.cntrs       = [self.cntrZeroVec] * self.numCntrs
        self.mode        = mode
        
        if (self.mode=='stat'):
            self.expSize = expSize
            self.calcParamsStat ()
        elif (mode=='dyn'):
            self.calcParamsDyn ()
        else:
            settings.error ('in SEAD.py: mode {} of SEAD does not exist' .format (self.mode))
             
    def rstCntr (self, cntrIdx=0):
        """
        """
        self.cntrs[cntrIdx] = self.cntrZeroVec
        
    def calcParamsStat (self):
        """
        Pre-compute the cntrs' parameters, in case of a static SEAD cntr 
        """
        if (self.expSize >= self.cntrSize):
            print ('error: for cntrSize={}, the maximal allowed expSize is {}' .format (self.cntrSize, self.expSize))
            exit  ()
        self.cntrMaxVec = '1' * self.cntrSize
        self.mantSize   = self.cntrSize - self.expSize
        self.expMaxVal  = 2**self.expSize - 1
        self.calcOffsets ()
        self.cntrMaxVal = self.valOf (mantVal=2**self.mantSize-1, expVal=self.expMaxVal)               
   
    def calcParamsDyn (self):
        """
        Pre-compute the cntrs' parameters, in case of a dynamic SEAD cntr 
        """
        self.cntrMaxVec = '1' * (self.cntrSize-2) + '0' + '1'
        self.expMaxVal  = self.cntrSize-2
        self.calcOffsets ()
        self.cntrMaxVal = self.valOf (mantVal=1, expVal=self.expMaxVal)                 
        
    def cntr2num (self, cntr, verbose=None):
        """
        Convert a counter, given as a binary vector (e.g., "11110"), to an integer num.
        Input: cntr, , given as a binary vector (e.g., "11110").
        Output: integer.
        """
        
        if (verbose!=None): #if a new verbose was given, it overrides the current verbose
            self.verbose = verbose
        if (len(cntr) != self.cntrSize): # if the cntr's size differs from the default, we have to update the basic params
            print ('the size of the given counter is {} while CntrMaster was initialized with cntrSize={}.' .format (len(cntr), self.cntrSize))
            print ('Please initialize a cntr with the correct len.')
            exit ()        
        if (self.mode=='stat'):
            expVec  = cntr[:self.expSize]
            mantVec = cntr[self.expSize:]
            if (settings.VERBOSE_COUT_CNTRLINE in self.verbose):
                expVal  = int (expVec, base=2)
                mantVal = int (mantVec, base=2)
                cntrVal = self.valOf (expVal=expVal, mantVal=mantVal)
                self.printCntrLine (cntr=cntr, expVec=expVec, expVal=expVal, mantVec=mantVec, mantVal=mantVal, cntrVal=cntrVal)
            return self.valOf (expVal=int (expVec, base=2), mantVal=int (mantVec, base=2))
        elif (self.mode=='dyn'):
            expSize = settings.idxOfLeftmostZero (ar=cntr, maxIdx=self.cntrSize-2)
            expVal  = expSize
            expVec  = cntr[:expSize]
            mantVec = cntr[expSize+1:]
            if (settings.VERBOSE_COUT_CNTRLINE in self.verbose):
                mantVal = int (mantVec, base=2)
                cntrVal = self.valOf (expVal=expVal, mantVal=mantVal)
                self.printCntrLine (cntr=cntr, expVec=expVec, expVal=expVal, mantVec=mantVec, mantVal=mantVal, cntrVal=cntrVal)
            return self.valOf (expVal=expVal, mantVal=int (mantVec, base=2))

    def cntr2numDyn (self, cntr):
        """
        Convert a dynamic counter, given as a binary vector (e.g., "11110"), to an integer num.
        Input: cntr, , given as a binary vector (e.g., "11110").
        Output: integer.
        """
        return 
    
    # def calcNprintCntr (self, cntr, expVec, mantVec):
    #     """
    #     Perform the final calculation (which are common for F2P, F3P modes); calculate the counter; and print the res (if requested by the user's verbose).
    #     Returns the value of the cntr (as int). 
    #     """
    
    # def idxOfLeftmostZero (self, cntr, maxIdx):
    #     """
    #     if the index of the leftmost 0 in the cntr >= maxIdx, return maxIdx.
    #     else, return the index of the leftmost 0 in the cntr.
    #     """ 
    #     if (cntr == '1' * len(cntr)): 
    #         return maxIdx
    #     return min (cntr.index('0'), maxIdx)
    
    def queryCntr (self, cntrIdx=0) -> dict:
        """
        Query a cntr.
        Input: 
        cntrIdx - the counter's index. 
        Output:
        cntrDic: a dictionary, where: 
            - cntrDict['cntrVec'] is the counter's binary representation; cntrDict['val'] is its value.        
        """
        settings.checkCntrIdx (cntrIdx=cntrIdx, numCntrs=self.numCntrs, cntrType='SEAD')
        return {'cntrVec' : self.cntrs[cntrIdx], 'val' : self.cntr2num(self.cntrs[cntrIdx])}    
        
    # def setMantExpDelta (self, cntrIdx):
    #     """
    #     Set/update the following fields of self:
    #     mantVec, mantVal, expVec, expVal, delta.
    #     The fields are set according to those of self.cntrs[cntrIdx]
    #     Currently unused.
    #     """
    #     self.expVec  = self.getExpVecStat  (cntrIdx)
    #     self.mantVec = self.getMantVecStat (cntrIdx)            
    #     self.expVal  = int (self.expVec, base=2) #self.getExpValStat  (cntrIdx)            
    #     self.mantVal = int (self.mantVec, base=2) #self.getExpValStat  (cntrIdx)
    #     self.cntrVal = self.cntr2num(self.cntrs[cntrIdx])
        
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
        self.targetVal = (self.cntr2num (self.cntrs[cntrIdx]) * factor) if mult else (self.cntr2num (self.cntrs[cntrIdx]) + factor)
        if (self.targetVal >= self.cntrMaxVal):
            self.cntrs[cntrIdx] = self.cntrMaxVec
            return {'cntrVec' : self.cntrs[cntrIdx], 'val' : self.cntrMaxVal}
        elif (self.targetVal <= 0):
            self.cntrs[cntrIdx] = self.cntrZeroVec
            return {'cntrVec' : self.cntrs[cntrIdx], 'val' : 0}
        # cntrRecord = 
        return self.incCntrStat (cntrIdx=cntrIdx) if (self.mode=='stat') else self.incCntrDyn (cntrIdx=cntrIdx) #$$$ 

    def incCntrDyn (self, cntrIdx=0):
        
        offset  = max ([offset for offset in self.offsetOfExpVal if offset<=self.targetVal]) # find the maximal offset which is <= targetVal
        expVal  = list(self.offsetOfExpVal).index(offset) # expVal is the index of this offset in self.offsetOfExpVal 
        expSize = expVal
        mantVal = math.floor (float(self.targetVal-offset)/float(2**expVal))
        mantSize = self.cntrSize - expSize - 1
        self.cntrs[cntrIdx] = '1' * expVal + '0' + np.binary_repr(mantVal, mantSize)
        cntrVal = self.cntr2num(self.cntrs[cntrIdx])

        if (cntrVal==self.targetVal): # does the cntr accurately represent the target value?
            return {'cntrVec' : self.cntrs[cntrIdx], 'val' : cntrVal} # yep --> return the counter, and its value

        # now we know that the counter found is < the target value
        if (mantVal < 2**mantSize-1): # can we further increment the mantissa w/o o/f?
            cntrpp    = '1' * expSize + '0' + np.binary_repr (mantVal+1, mantSize)
        else:  # need to increase the exponent (and thus, also its size)
            cntrpp    = '1' * (expVal+1) + '0' * mantSize # need to decrement the mantissa field size.
        cntrppVal = self.cntr2num (cntrpp)

        probOfFurtherInc = float (self.targetVal - cntrVal) / float (cntrppVal - cntrVal)
        if (random.random() < probOfFurtherInc):
            self.cntrs[cntrIdx] = cntrpp
            return {'cntrVec' : self.cntrs[cntrIdx], 'val' : cntrppVal}
        return {'cntrVec' : self.cntrs[cntrIdx], 'val' : cntrVal} 
                
    def incCntrStat (self, cntrIdx=0):
               
        offset  = max ([offset for offset in self.offsetOfExpVal if offset<=self.targetVal]) # find the maximal offset which is <= targetVal
        expVal  = list(self.offsetOfExpVal).index(offset) # expVal is the index of this offset in self.offsetOfExpVal 
        mantVal = math.floor (float(self.targetVal-offset)/float(2**expVal))

        self.cntrs[cntrIdx] = np.binary_repr(expVal, self.expSize) + np.binary_repr(mantVal, self.mantSize)
        cntrVal = self.cntr2num(self.cntrs[cntrIdx])

        if (cntrVal==self.targetVal or cntrVal==self.cntrMaxVal): # does the cntr accurately represent the target value?
            return {'cntrVec' : self.cntrs[cntrIdx], 'val' : cntrVal} # yep --> return the counter, and its value

        # now we know that the counter found is < the target value
        if (mantVal < 2**self.mantSize-1): # can we further increment the mantissa w/o o/f?
            cntrpp    = np.binary_repr(expVal,   self.expSize) + np.binary_repr (mantVal+1, self.mantSize)
        else:  # need to increase the exponent
            cntrpp    = np.binary_repr(expVal+1, self.expSize) + '0' * self.mantSize # need to decrement the mantissa field size.
        cntrppVal = self.cntr2num (cntrpp)

        probOfFurtherInc = float (self.targetVal - cntrVal) / float (cntrppVal - cntrVal)
        if (random.random() < probOfFurtherInc): # further increment?
            self.cntrs[cntrIdx] = cntrpp         # Yep
            return {'cntrVec' : self.cntrs[cntrIdx], 'val' : cntrppVal}
        return {'cntrVec' : self.cntrs[cntrIdx], 'val' : cntrVal} 

def printAllVals (cntrSize=4, expSize=1, mode='stat', verbose=[]):
    """
    Loop over all the binary combinations of the given counter size. 
    For each combination, print to file the respective counter, and its value. 
    The prints are sorted in an increasing order of values.
    """
    if (mode not in ['stat', 'dyn']):
        print ('sorry, mode {} that you chose is not supported yet' .format (mode))
        exit ()
    print ('running printAllVals. mode={}' .format (mode))
    listOfVals = []
    myCntrMaster = CntrMaster(cntrSize=cntrSize, expSize=expSize, mode=mode, verbose=verbose)
    if (mode=='stat'):
        for i in range (2**cntrSize):
            cntr = np.binary_repr(i, cntrSize) 
            listOfVals.append ({'cntrVec' : cntr, 'val' : myCntrMaster.cntr2num(cntr)})
    else:
        for i in range (2**cntrSize-2):
            cntr = np.binary_repr(i, cntrSize) 
            listOfVals.append ({'cntrVec' : cntr, 'val' : myCntrMaster.cntr2num(cntr)}) 
    listOfVals = sorted (listOfVals, key=lambda item : item['val'])

    if (settings.VERBOSE_RES in verbose):
        myCntrMaster.mode       = mode
        myCntrMaster.cntrSize   = cntrSize
        myCntrMaster.expSize    = expSize
        outputFile    = open ('../res/{}.res' .format (myCntrMaster.genSettingsStr()), 'w')
        for item in listOfVals:
            printf (outputFile, '{}={}\n' .format (item['cntrVec'], item['val']))
    print ('cntrMaxVal={}' .format (myCntrMaster.cntrMaxVal))

def printAllCntrMaxVals (mode='stat', cntrSizes=[], expSizes=None, verbose=[settings.VERBOSE_RES]):
    """
    print the maximum value a cntr reach for several "configurations" -- namely, all combinations of cntrSize and hyperSize. 
    """

    if (settings.VERBOSE_RES in verbose):
        outputFile    = open ('../res/cntrMaxVals.txt', 'a')
    if (mode=='stat'):
        for cntrSize in cntrSizes:
            expSizes = expSizes if (expSizes!=None) else range (1, cntrSize)
            for expSize in expSizes:
                # print ('cntrSize={}, expSize={}' .format (cntrSize, expSize))
                myCntrMaster = CntrMaster (mode='stat', cntrSize=cntrSize, expSize=expSize)
                printf (outputFile, '{} cntrMaxVal={:.0f}\n' .format (myCntrMaster.genSettingsStr (mode='stat', cntrSize=cntrSize, expSize=expSize), myCntrMaster.cntrMaxVal))
    elif (mode=='dyn'):
        for cntrSize in cntrSizes:
            # print ('cntrSize={}'.format (cntrSize))
            myCntrMaster = CntrMaster(mode='dyn', cntrSize=cntrSize)
            printf (outputFile, '{} cntrMaxVal={:.0f}\n' .format (myCntrMaster.genSettingsStr (mode='dyn', cntrSize=cntrSize, expSize=0), myCntrMaster.cntrMaxVal))
    else:
        print ('Sorry, mode {} is not supported yet' .format (mode))

def checkTimes ():
    """
    check which code style is faster.
    The tests show that shift is slightly slower than mult.
    """
    cntrSize = 16
    mantVal = 1
    
    startTime = time.time ()
    for _ in range (50):
        for i in range (2**cntrSize):
            cntr = np.binary_repr(i, cntrSize) 
            for expSize in range (1, 4):
                expVal  = int (cntr[:expSize], base=2)
                cntrVal = mantVal*2**expVal
    print ('t by mult={}' .format (time.time()-startTime))

    startTime = time.time ()
    for _ in range (50):
        for i in range (2**cntrSize):
            cntr = np.binary_repr(i, cntrSize) 
            for expSize in range (1, 4):
                expVal  = int (cntr[:expSize], base=2)
                cntrValByShift = mantVal << expVal
    print ('t by shift={}' .format (time.time()-startTime))

