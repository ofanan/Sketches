# This class implements the CEDAR counter, with a support for down-sampling.
# CEDAR is described in the paper: "Estimators also need shared values to grow together", Tsidon, Erez and Hanniel, Iddo and Keslassy, Isaac, Infocom'15.
import random, math, numpy as np

from printf import printf
import settings, CEDAR
from settings import *

class CntrMaster (CEDAR.CntrMaster):
    """
    Generate, check and parse counters
    """

    def incCntrBy1GetVal (self, cntrIdx=0):
        """
        """
        if (self.cntrs[cntrIdx] == self.numEstimators-1): # reached the largest estimator --> cannot further inc
            if VERBOSE_LOG_DWN_SMPL in self.verbose:
                if self.numCntrs<10:
                    printf (self.logFile, f'b4 upScaling:\n')
                    self.printAllCntrs (self.logFile)
                else:
                    printf (self.logFile, 'cntrVal={:.0f}. upScaling.\n' .format (
                                           self.cntr2num(self.cntrs[cntrIdx])))
            self.upScale ()
            if VERBOSE_LOG_DWN_SMPL in self.verbose:
                if self.numCntrs<10:
                    printf (self.logFile, f'\nafter upScaling:\n')
                    self.printAllCntrs (self.logFile)
            return self.estimators[self.cntrs[cntrIdx]]
        if random.random() < 1/self.diffs[self.cntrs[cntrIdx]]:
            self.cntrs[cntrIdx] += 1
        return self.estimators[self.cntrs[cntrIdx]]

    def upScale (self):
        """
        Allow down-sampling:
        - Calculate a new "delta" parameter that allows reaching a higher cntrMaxVal.
        - Calculate new cntrs' value to keep roughly the estimation as before the upscale.  
        """
        prevEstimators = self.estimators.copy()
        # prevDelta, prevDiffs, prevEstimators = self.delta, self.diffs.copy(), self.estimators.copy()
        prevCntrMaxVal   = self.cntrMaxVal 
        self.cntrMaxVal *= 2
        
        self.findMinDeltaByMaxVal (
            targetMaxVal    = self.cntrMaxVal,
            deltaLo         = 0.00001,
            deltaHi         = 0.4
        )                
                
        if VERBOSE_DEBUG in self.verbose:
            self.cntrs = [i for i in range(self.numCntrs)]
                        
        for cntrIdx in range(self.numCntrs):
            orgVal = prevEstimators[self.cntrs[cntrIdx]]
            newEstIdx = 0
            while self.estimators[newEstIdx] < orgVal:
                newEstIdx += 1
            # Now we know that self.estimators[newEstIdx] >= orgVal
            if self.estimators[newEstIdx]==orgVal:
                self.cntrs[cntrIdx] = newEstIdx
                if VERBOSE_DEBUG in self.verbose:
                    printf (self.logFile, 'orgVal=val={:.1f}\n' .format (orgVal))
                continue
            if random.random() < (orgVal-self.estimators[newEstIdx-1])/(self.estimators[newEstIdx]-self.estimators[newEstIdx-1]):
                self.cntrs[cntrIdx] = newEstIdx
            else:
                self.cntrs[cntrIdx] = newEstIdx-1
            if VERBOSE_DEBUG in self.verbose:
                floorVal = self.estimators[newEstIdx-1]
                ceilVal  = self.estimators[newEstIdx]
                printf (self.logFile, 'orgVal={:.1f}, floorVal={:.1f}, ceilVal={:.1f}, val={:.1f}\n' 
                       .format (orgVal, floorVal, ceilVal, self.estimators[self.cntrs[cntrIdx]]))
        
        if VERBOSE_DEBUG in self.verbose:
            printf (self.logFile, 'Printing all estimators\n')
            for estimator in self.estimators:
                printf (self.logFile, '{:.1f} ' .format(estimator)) 
        
# def printAllVals(cntrSize=8, delta=None, cntrMaxVal=None, verbose=[]):
#     """
#     Loop over all the binary combinations of the given counter size.
#     For each combination, print to file the respective counter, and its value.
#     The prints are sorted in an increasing order of values.
#     """
#     listOfVals = []
#     myCntrMaster = CntrMaster(cntrSize=cntrSize, delta=delta, cntrMaxVal=cntrMaxVal, numCntrs=1)
#     for num in range(2 ** cntrSize):
#         val = myCntrMaster.cntrInt2num(num)
#         listOfVals.append ({'cntrVec' : np.binary_repr(num, cntrSize), 'val' : val})
#
#
#     if settings.VERBOSE_RES in verbose:
#         outputFile = open('../res/single_cntr_log_files/{}.res'.format(myCntrMaster.genSettingsStr()), 'w')
#         for item in listOfVals:
#             printf(outputFile, '{}={:.1f}\n'.format(item['cntrVec'], item['val']))

# \frac{\left(\left(1+2\cdot \:\:x^2\right)^L-1\right)}{2x^2}\left(1+x^2\right)


# myCntrMaster = CntrMaster (
#     numCntrs    = 2**6,
#     cntrSize    = 6, 
#     cntrMaxVal  = 1000,
#     verbose     = [VERBOSE_DEBUG]
# ) 
# logFile = open (f'../res/log_files/{myCntrMaster.genSettingsStr()}.log', 'w')
# myCntrMaster.setLogFile (logFile)
# myCntrMaster.upScale()
