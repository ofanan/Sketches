# Additive Error Estimator, descirbed in the paper:
# [AEE] "Faster and more accurate measurement through additive-error counters", Ran Ben Basat; Gil Einziger; Michael Mitzenmacher; Shay Vargaftik.
# https://ieeexplore.ieee.org/document/9155340
# The probabilistic increment variable, self.P, is set only once, at the generation of the counters.
# *** Hence, this version of AEE is without down-sampling (unless someone updates self.p).   
import math, time, random
from printf import printf
import settings, Cntr
import numpy as np
from settings import VERBOSE_DEBUG, VERBOSE_LOG, VERBOSE_LOG_DWN_SMPL
from settings import warning, error

class CntrMaster (Cntr.CntrMaster):
    """
    Generate, check and parse counters
    """

    # Generates a strings that details the counter's settings (param vals).    
    genSettingsStr = lambda self : 'AEE_n{}' .format (self.cntrSize)
       
    cntr2num    = lambda self, cntr : cntr / self.p # Given the counter (as a binary vector string) return the value it represents
    
    setP        = lambda self : self.cntrMaxVec / self.cntrMaxVal # Set the value of the increment prob', p. 

    def __init__ (
            self, 
            cntrMaxVal    = 30, # Denoted N in [AEE] 
            cntrSize      = 4, # num of bits in each counter. 
            numCntrs      = 1, # number of counters in the array.
            verbose       = [], # determines which outputs would be written to .log/.res/.pcl/debug files, as detailed in settings.py. 
        ):

        super(CntrMaster, self).__init__ (
            cntrSize    = cntrSize, 
            numCntrs    = numCntrs,
            verbose     = verbose
        )
        self.cntrMaxVal  = cntrMaxVal # The maximal value that can be coded by the estimators
        if self.cntrSize<=8:
            self.cntrs       = np.zeros (self.numCntrs, dtype=np.int8)
        elif self.cntrSize<=16:
            self.cntrs       = np.zeros (self.numCntrs, dtype=np.int16)
        else:
            self.cntrs       = np.zeros (self.numCntrs, dtype=np.int32)
        self.cntrMaxVec  = 2**self.cntrSize - 1 # The maximal value of a self.cntrSize-bits integer estimator.
        self.cntrZeroVec = 0 
        self.p           = self.setP ()
            
    def incCntrBy1 (
            self, 
            cntrIdx  = 0, # idx of the concrete counter to increment in the array 
            forceInc = False
        ) -> dict: # If forceInc==True, increment the counter. Else, inc the counter w.p. p 
        """
        Perform probabilistic-Increment of the counter to the closest higher value.        
        Output:
        cntrDict: a dictionary representing the modified counter where: 
            - cntrDict['cntrVec'] is the counter's binary representation; cntrDict['val'] is its value.
        """
        settings.checkCntrIdx (cntrIdx=cntrIdx, numCntrs=self.numCntrs, cntrType='AEE')
        if self.cntrs[cntrIdx]!=self.cntrMaxVec and (forceInc or random.random() < self.p):
            self.cntrs[cntrIdx] += 1
        return {'cntrVec' : self.cntrs[cntrIdx], 'val' : self.cntr2num(self.cntrs[cntrIdx])}

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
        if self.cntrs[cntrIdx]!=self.cntrMaxVec and (forceInc or random.random() < self.p):
            self.cntrs[cntrIdx] += 1
        return self.cntr2num(self.cntrs[cntrIdx])


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
        outputFile    = open ('../res/single_cntr_log_files/{}.res' .format (myCntrMaster.genSettingsStr()), 'w')
    
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
      