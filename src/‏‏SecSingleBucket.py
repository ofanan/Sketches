# This file is obsolete. SEC (Shared Exponent Counters) are currently implemented in SEC.py.
import math, time, random
from printf import printf
import settings
import numpy as np

class SecSingleBucket (object):
    """
    Generate, check and parse counters
    """

    # Generates a strings that details the counter's settings (param vals).    
    genSettingsStr = lambda self : f'SEC_n{self.cntrSize}_e{self.exp}'
       
    # Given a counter, return the number it represents
    cntr2num       = lambda self, cntr : cntr << self.exp

    # Given a counter, return a dictionary cntrDict, where: cntrDict['cntrVec'] is the counter's binary representation; cntrDict['val'] is its value.            
    cntr2cntrDict = lambda self, cntr : {'cntrVec' : np.binary_repr(cntr, width=self.cntrSize), 'val' : self.cntr2num(cntr)}
    
    def __init__ (self, 
                  cntrSize      = 4, # num of bits in each counter. 
                  numCntrs      = 2, # number of counters in the array.
                  verbose       = [], # determines which outputs would be written to .log/.res/.pcl/debug files, as detailed in settings.py. 
                  ):

        if (cntrSize<1):
            settings.error ('in SEC: cntrSize requested is {}. However, cntrSize should be at least 2.' .format (cntrSize))
            
        self.cntrSize   = int(cntrSize)
        self.numCntrs   = int(numCntrs)
        self.verbose    = verbose
        self.cntrMaxVec = (1 << self.cntrSize) - 1
        self.exp        = 0
        self.sampleProb = 1
        if self.cntrSize<=8:
            self.cntrs      = np.zeros(self.numCntrs, dtype='int8') 
        elif self.cntrSize<=16:
            self.cntrs      = np.zeros(self.numCntrs, dtype='int16') 
        else:
            self.cntrs      = np.zeros(self.numCntrs, dtype='int32')             
        
    def rstCntr (self, cntrIdx=0):
        """
        """
        self.cntrs[cntrIdx] = 0

    def queryCntr (self, cntrIdx=0) -> dict:
        """
        Query a cntr.
        Input: 
        cntrIdx - the counter's index. 
        Output:
        cntrDic: a dictionary, where: 
            - cntrDict['cntrVec'] is the counter's binary representation; cntrDict['val'] is its value.        
        """
        settings.checkCntrIdx (cntrIdx=cntrIdx, numCntrs=self.numCntrs, cntrType='SEC')
        return self.cntr2cntrDict(self.cntrs[cntrIdx])           

    def queryCntrVal (self, cntrIdx=0) -> int:
        """
        Query a cntr.
        Input: 
        cntrIdx - the counter's index. 
        Output:
        The value that the counter represents (as int/FP).
        """
        return self.cntr2num(self.cntrs[cntrIdx])                   
        
    def incExp (self):
        """
        Increment the exponent by 1 and divide all the counters by 2.
        """
        self.exp        += 1
        self.sampleProb  = 1/(2**self.exp)
        self.cntrs       = [cntr//2 for cntr in self.cntrs]
    
    def incCntrBy1GetVal (self, 
                          cntrIdx  = 0, # idx of the concrete counter to increment in the array 
                          ) -> int: # If forceInc==True, increment the counter. Else, inc the counter w.p. p 
        """
        Increment the counter to the closest higher value.        
        Output:
        cntrDict: a dictionary representing the modified counter where: 
            - cntrDict['cntrVec'] is the counter's binary representation; cntrDict['val'] is its value.
        """
        if self.exp==0 or (random.random() < self.sampleProb):
            if (self.cntrs[cntrIdx]==self.cntrMaxVec):
                self.cntrs[cntrIdx] += 1
                self.incExp()
            else:
                self.cntrs[cntrIdx] += 1
        return self.cntr2num(self.cntrs[cntrIdx])

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
        """
        if mult or (factor!=1):
            settings.error ('Sorry, .incCntr() is currently implemented only when mult==True and factor=1.')
        return self.incCntrBy1 (cntrIdx=cntrIdx) 

def printAllVals (cntrSize=4, verbose=[], cntMax=100):
    """
    Loop over all the binary combinations of the given counter size. 
    For each combination, print to file the respective counter, and its value. 
    The prints are sorted in an increasing order of values.
    """
    myCntrMaster = CntrMaster(cntrSize=cntrSize)

    if (settings.VERBOSE_RES in verbose):
        outputFile    = open ('../res/{}.res' .format (myCntrMaster.genSettingsStr()), 'w')
    
    print ('running printAllVals')
    listOfVals = [0]
    for i in range (cntMax):
        listOfVals.append (myCntrMaster.incCntrBy1GetVal())
    
    if (settings.VERBOSE_RES in verbose):
        for item in listOfVals:
            printf (outputFile, f'{item}\n')
    
    if (settings.VERBOSE_PCL in verbose):
        with open('../res/pcl_files/{}.pcl' .format (myCntrMaster.genSettingsStr()), 'wb') as pclOutputFile:
            pickle.dump(listOfVals, pclOutputFile) 
      
printAllVals(cntrSize=4, verbose=[settings.VERBOSE_RES])
