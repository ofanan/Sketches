# Additive Error Estimator, descirbed in the paper:
# [AEE] "Faster and more accurate measurement through additive-error counters", Ran Ben Basat; Gil Einziger; Michael Mitzenmacher; Shay Vargaftik.
# https://ieeexplore.ieee.org/document/9155340 
import math, time, random
from printf import printf
import settings
import numpy as np

class CntrMaster (object):
    """
    Generate, check and parse counters
    """

    # Generates a strings that details the counter's settings (param vals).    
    genSettingsStr = lambda self : 'AEE_n{}' .format (self.cntrSize)
       
    cntr2num    = lambda self, cntr : (int(cntr, base=2) / self.p) # Given the counter (as a binary vector string) return the value it represents
    
    setP        = lambda self : self.cntrMaxVecInt / self.cntrMaxVal # Set the value of the increment prob', p. 

    def __init__ (self, 
                  cntrMaxVal    = 30, # Denoted N in [AEE] 
                  cntrSize      = 4, # num of bits in each counter. 
                  numCntrs      = 1, # number of counters in the array.
                  epsilon       = 1,
                  delta         = 0.01, 
                  verbose       = [], # determines which outputs would be written to .log/.res/.pcl/debug files, as detailed in settings.py. 
                  ):

        if (cntrSize<2):
            print ('error: cntrSize requested is {}. However, cntrSize should be at least 2.' .format (cntrSize))
            exit ()
            
        self.cntrMaxVal     = cntrMaxVal   # Denoted N in [AEE]
        self.cntrSize       = int(cntrSize)
        self.numCntrs       = int(numCntrs)
        self.verbose        = verbose
        self.cntrZeroVec    = '0' * self.cntrSize
        self.cntrs          = [self.cntrZeroVec for i in range (self.numCntrs)]
        self.cntrMaxVec     = '1' * self.cntrSize
        self.cntrMaxVecInt  = (1 << self.cntrSize) - 1
        self.cntrZero       = 0
        self.epsilon        = epsilon
        self.delta          = delta
        self.p              = self.setP ()
            
    def rstCntr (self, cntrIdx=0):
        """
        """
        self.cntrs[cntrIdx] = self.cntrZeroVec

    def queryCntr (self, cntrIdx=0) -> dict:
        """
        Query a cntr.
        Input: 
        cntrIdx - the counter's index. 
        Output:
        cntrDic: a dictionary, where: 
            - cntrDict['cntrVec'] is the counter's binary representation; cntrDict['val'] is its value.        
        """
        settings.checkCntrIdx (cntrIdx=cntrIdx, numCntrs=self.numCntrs, cntrType='AEE')
        return {'cntrVec' : self.cntrs[cntrIdx], 'val' : self.cntr2num(self.cntrs[cntrIdx])}           

    def incCntrBy1 (self, 
                    cntrIdx  = 0, # idx of the concrete counter to increment in the array 
                    forceInc = False) -> dict: # If forceInc==True, increment the counter. Else, inc the counter w.p. p 
        """
        Increment the counter to the closest higher value.        
        Output:
        cntrDict: a dictionary representing the modified counter where: 
            - cntrDict['cntrVec'] is the counter's binary representation; cntrDict['val'] is its value.
        """
        settings.checkCntrIdx (cntrIdx=cntrIdx, numCntrs=self.numCntrs, cntrType='AEE')
        if self.cntrs[cntrIdx]!=self.cntrMaxVec and (forceInc or random.random() < self.p):
            self.cntrs[cntrIdx] = np.binary_repr (int (self.cntrs[cntrIdx], 2) + 1, self.cntrSize)
        return {'cntrVec' : self.cntrs[cntrIdx], 'val' : self.cntr2num(self.cntrs[cntrIdx])}


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
        settings.checkCntrIdx (cntrIdx=cntrIdx, numCntrs=self.numCntrs, cntrType='AEE')
        if mult or (factor!=1):
            settings.error ('Sorry, AEE.incCntr() is currently implemented only when mult==True and factor=1.')
        return self.incCntrBy1 (cntrIdx=cntrIdx) 

def printAllVals (cntrSize=4, cntrMaxVal=100, verbose=[]):
    """
    Loop over all the binary combinations of the given counter size. 
    For each combination, print to file the respective counter, and its value. 
    The prints are sorted in an increasing order of values.
    """
    if cntrMaxVal < 2**cntrSize:
        settings.error (f'cntrMaxVal={cntrMaxVal} while max accurately representable value for {cntrSize}-bit counter is {2**cntrSize-1}')
    myCntrMaster = CntrMaster(cntrSize=cntrSize, cntrMaxVal=cntrMaxVal)

    if (settings.VERBOSE_RES in verbose):
        outputFile    = open ('../res/{}.res' .format (myCntrMaster.genSettingsStr()), 'w')
    
    print ('running printAllVals')
    listOfVals = []
    for i in range (2**cntrSize):
        cntr = np.binary_repr(i, cntrSize) 
        listOfVals.append ({'cntrVec' : cntr, 'val' : myCntrMaster.cntr2num(cntr)})

    if (settings.VERBOSE_RES in verbose):
        for item in listOfVals:
            printf (outputFile, '{}={:.2f}\n' .format (item['cntrVec'], item['val']))
    
    if (settings.VERBOSE_PCL in verbose):
        with open('../res/pcl_files/{}.pcl' .format (myCntrMaster.genSettingsStr()), 'wb') as pclOutputFile:
            pickle.dump(listOfVals, pclOutputFile) 
      

# printAllVals(cntrSize=8, cntrMaxVal=1488888, verbose=[settings.VERBOSE_RES])