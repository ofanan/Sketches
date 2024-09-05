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
        
        deltaHi = 0.7 if self.cntrSize<8 else 0.4
        self.findMinDeltaByMaxVal (
            targetMaxVal    = self.cntrMaxVal,
            deltaLo         = 0.00001,
            deltaHi         = deltaHi
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
        