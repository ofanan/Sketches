# Additive Error Estimator, descirbed in the paper:
# [AEE] "Faster and more accurate measurement through additive-error counters", Ran Ben Basat; Gil Einziger; Michael Mitzenmacher; Shay Vargaftik.
# https://ieeexplore.ieee.org/document/9155340
# The probabilistic increment variable, self.P, is set only once, at the generation of the counters.
# *** Hence, this version of AEE is without down-sampling (unless someone updates self.p).   
import math, time, random
from printf import printf
import settings, AEE
import numpy as np
from settings import *

class CntrMaster (AEE.CntrMaster):
    """
    Generate, check and parse counters
    """

    def incCntrBy1GetVal (
            self, 
            cntrIdx  = 0, # idx of the concrete counter to increment in the array 
            forceInc = False
        ) -> dict: # If forceInc==True, increment the counter. Else, inc the counter w.p. p 
        """
        Perform probabilistic-Increment of the counter to the closest higher value.        
        Output:
        The value after the operation. 
        """        
        if self.cntrs[cntrIdx]==self.cntrMaxVec: # Cntr is saturated --> need to down-sample
            if self.cntr2num(self.cntrs[cntrIdx])!=self.cntrMaxVec/self.p:
                error ('In AEE_ds.incCntrBy1GetVal(). Wrong CntrMaxVal. cntrVal={self.cntr2num(self.cntrs[cntrIdx])}self.cntr2num(self.cntrs[cntrIdx]), curCntrMaxVal={self.cntrMaxVec/self.p}')
            if VERBOSE_LOG_DWN_SMPL in self.verbose:
                if self.numCntrs<10:
                    printf (self.logFile, f'\nb4 upScaling:\n')
                    self.printAllCntrs (self.logFile)
                else:
                    printf (self.logFile, 'cntrVal={:.0f}. upScaling.\n' .format (self.cntrMaxVec/self.p))
            self.upScale ()
            if VERBOSE_LOG_DWN_SMPL in self.verbose:
                if self.numCntrs<10:
                    printf (self.logFile, f'\nafter upScaling:\n')
                    self.printAllCntrs (self.logFile)
        if random.random() < self.p: # Perform prob' increment
            self.cntrs[cntrIdx] += 1 
        return self.cntr2num(self.cntrs[cntrIdx])

    def upScale (self):
        """
        Allow down-sampling:
        - Half the values of all the counters.
        - Half self.p.
        """
        for i in range(self.numCntrs):
            if self.cntrs[i]%2==1: # The counter is odd - maybe we'll have to ceil the value after halving.
                self.cntrs[i] //= 2  
                if random.random() < 0.5:
                    self.cntrs[i] += 1
            else: # The counter is even - no need to ceil the value after halving.
                self.cntrs[i] //= 2  
        self.p /= 2

def printAllVals (cntrSize=4, cntrMaxVal=100, verbose=[]):
    """
    Loop over all the binary combinations of the given counter size. 
    For each combination, print to file the respective counter, and its value. 
    The prints are sorted in an increasing order of values.
    """
    if cntrMaxVal < 2**cntrSize:
        settings.error (f'cntrMaxVal={cntrMaxVal} while max accurately representable value for {cntrSize}-bit counter is {2**cntrSize-1}')
    myCntrMaster = CntrMaster(            
        cntrMaxVal    = cntrMaxVal, # Denoted N in [AEE] 
        cntrSize      = cntrSize, # num of bits in each counter. 
        numCntrs      = 1, # number of counters in the array.
    )

    if (settings.VERBOSE_RES in verbose):
        outputFile    = open ('../res/single_cntr_log_files/{}.res' .format (myCntrMaster.genSettingsStr()), 'w')
    
    print ('running printAllVals')
    listOfVals = []
    for i in range (2**cntrSize):
        listOfVals.append ({'cntrVec' : i, 'val' : myCntrMaster.cntr2num(i)})

    if (settings.VERBOSE_RES in verbose):
        for item in listOfVals:
            printf (outputFile, '{}={:.2f}\n' .format (item['cntrVec'], item['val']))
    
    if (settings.VERBOSE_PCL in verbose):
        with open('../res/pcl_files/{}.pcl' .format (myCntrMaster.genSettingsStr()), 'wb') as pclOutputFile:
            pickle.dump(listOfVals, pclOutputFile)

