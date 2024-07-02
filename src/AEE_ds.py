# Additive Error Estimator, descirbed in the paper:
# [AEE] "Faster and more accurate measurement through additive-error counters", Ran Ben Basat; Gil Einziger; Michael Mitzenmacher; Shay Vargaftik.
# https://ieeexplore.ieee.org/document/9155340
# The probabilistic increment variable, self.P, is set only once, at the generation of the counters.
# *** Hence, this version of AEE is without down-sampling (unless someone updates self.p).   
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
            forceInc = False
        ) -> dict: # If forceInc==True, increment the counter. Else, inc the counter w.p. p 
        """
        Perform probabilistic-Increment of the counter to the closest higher value.        
        Output:
        The value after the operation. 
        """        
        if self.cntrs[cntrIdx]==self.cntrMaxVec: # Cntr is saturated --> need to down-sample
            self.upScale ()
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
            if self.cntrs[i][-1]%2==1: # The counter is odd - maybe we'll have to ceil the value after halving.
                self.cntrs[i] //= 2  
                if random.random() < 0.5:
                    self.cntrs[i] += 1
            else: # The counter is even - no need to ceil the value after halving.
                self.cntrs[i] //= 2  
        self.p /= 2

aee_cntrMaster = CntrMaster (
    cntrMaxVal = 30, # Denoted N in [AEE] 
    cntrSize   = 4, # num of bits in each counter. 
)
# printAllVals(cntrSize=8, cntrMaxVal=1488888, verbose=[settings.VERBOSE_RES])