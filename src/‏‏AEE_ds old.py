# Additive Error Estimator, descirbed in the paper:
# [AEE] "Faster and more accurate measurement through additive-error counters", Ran Ben Basat; Gil Einziger; Michael Mitzenmacher; Shay Vargaftik.
# https://ieeexplore.ieee.org/document/9155340
# This version of AEE supports down-sampling.
# Namely, each time one tries to increment a saturated counter, all the counters are divided by a given factor (by default, 2), and the sampling prob' self.p is decreased accordingly. 
import math, time, random
from printf import printf
import settings, AEE
import numpy as np

class CntrMaster (AEE.CntrMaster):
    """
    Generate, check and parse counters
    """

    def incCntrBy1GetVal (
            self, 
            cntrIdx  = 0, # idx of the concrete counter to increment in the array 
        ) -> dict: # If forceInc==True, increment the counter. Else, inc the counter w.p. p 
        """
        Perform probabilistic-Increment of the counter to the closest higher value.        
        Output:
        The value after the operation. 
        """
        if random.random() < self.p: # Perform prob' increment
            if self.cntrs[cntrIdx]==self.cntrMaxVec: # Cntr is saturated --> need to down-sample
                for i in range(self.numCntrs):
                    if self.cntrs[i][-1]=='1': # The counter's LSB is -1
                        self.cntrs[i] = '0' + self.cntrs[i][0:-1] # mantVec >> 1 # divide the mantissa by 2 (by right-shift) 
                            if random.random() < 0.5:
                                self.cntrs[cntrIdx] = np.binary_repr (int (self.cntrs[cntrIdx], 2) + 1, self.cntrSize)
                             
                    mantVec = '0' + mantVec[0:-1] # mantVec >> 1 # divide the mantissa by 2 (by right-shift) 
                    
            if self.cntrs
            self.cntrs[cntrIdx] = np.binary_repr (int (self.cntrs[cntrIdx], 2) + 1, self.cntrSize)
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
      

aee_cntrMaster = CntrMaster (
    cntrMaxVal = 30, # Denoted N in [AEE] 
    cntrSize   = 4, # num of bits in each counter. 
)
# printAllVals(cntrSize=8, cntrMaxVal=1488888, verbose=[settings.VERBOSE_RES])